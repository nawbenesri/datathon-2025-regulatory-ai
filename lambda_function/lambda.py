"""
AWS Lambda function: Analyze regulation text with Bedrock (Claude 3),
retrieve 10-K snippets from Amazon Kendra, and synthesize a final JSON report.

Steps:
1) Use Claude 3 Sonnet to extract top-3 financial risk themes (keywords).
2) For each theme, query Kendra (Index ID from env or default) for relevant 10-K snippets.
3) Use Claude 3 Haiku to synthesize themes + snippets into a JSON report of top 5 exposed companies.
"""

import os
import json
import re
from typing import Dict, List, Any
import io

import boto3


# ---- Configuration ----
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
KENDRA_INDEX_ID = os.environ.get("KENDRA_INDEX_ID", "ffb5a04f-0e32-4c88-8c24-8beb1fc6a53f")

# Claude 3 model IDs (via Amazon Bedrock)
MODEL_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
S3_BUCKET = os.environ.get("S3_BUCKET", "regai-datathon-10k-reports-2025")


# ---- Clients ----
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
kendra = boto3.client("kendra", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)


# ---- Utilities ----
def _extract_json(text: str) -> Dict[str, Any]:
	"""Extract a JSON object from a model response (handles surrounding text/fences)."""
	if not text:
		return {}

	# Strip fenced code blocks if present
	m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.I)
	if m:
		text = m.group(1)

	# Greedy match last JSON object in text
	m2 = re.search(r"\{[\s\S]*\}\s*$", text)
	raw = m2.group(0) if m2 else text
	try:
		return json.loads(raw)
	except Exception:
		# Try trimming head/tail noise
		try:
			start = raw.find("{")
			end = raw.rfind("}")
			if start >= 0 and end > start:
				return json.loads(raw[start : end + 1])
		except Exception:
			pass
	return {"raw_response": text, "error": "json_parse_failed"}


def _invoke_bedrock_json(model_id: str, user_text: str, system_text: str = "",
						  max_tokens: int = 1500, temperature: float = 0.2) -> Dict[str, Any]:
	"""Invoke an Anthropic Claude 3 model via Bedrock and return parsed JSON."""
	body = {
		"anthropic_version": "bedrock-2023-05-31",
		"max_tokens": max_tokens,
		"temperature": temperature,
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": user_text}
				],
			}
		],
	}
	if system_text:
		body["system"] = [{"type": "text", "text": system_text}]

	resp = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
	data = json.loads(resp["body"].read())
	try:
		text = data["content"][0]["text"]
	except Exception:
		text = json.dumps(data)
	return _extract_json(text)


# ---- Step 1: Extract top-3 risk themes ----
def extract_risk_themes(regulation_text: str) -> List[str]:
	system = (
		"You are a senior financial regulation analyst for public equities. "
		"Return ONLY strict JSON."
	)
	user = f"""
Analyze the regulation text and extract the TOP 3 financial risk themes as concise finance-specific keywords, focusing on:
- regulatory compliance costs, reporting burden, and penalties
- tariffs/trade restrictions, export controls, sanctions
- taxation (minimum tax, surtax, digital services tax), subsidies, incentives
- supply chain disruptions and country/region dependencies
- product/segment exposure (e.g., semiconductors, pharma, payments)

Return STRICT JSON with this EXACT schema:
{{
	"themes": ["keyword1", "keyword2", "keyword3"]
}}

Text:
{regulation_text[:8000]}
"""
	result = _invoke_bedrock_json(MODEL_SONNET, user, system_text=system, max_tokens=600, temperature=0.1)
	themes = result.get("themes", [])
	# Normalize and trim
	themes = [str(t).strip() for t in themes if str(t).strip()]
	return themes[:3]


# ---- Step 2: Query Amazon Kendra for 10-K snippets ----
def _build_attribute_filter(filters: Dict[str, Any]) -> Dict[str, Any]:
	"""Build a simple Kendra AttributeFilter from provided filter dict.
	Supported keys: filing_type (str), ticker (str), fiscal_year (int).
	"""
	if not filters:
		return {}
	and_filters: List[Dict[str, Any]] = []
	if isinstance(filters.get("filing_type"), str):
		and_filters.append({
			"EqualsTo": {
				"Key": "filing_type",
				"Value": {"StringValue": filters["filing_type"]}
			}
		})
	if isinstance(filters.get("ticker"), str):
		and_filters.append({
			"EqualsTo": {
				"Key": "ticker",
				"Value": {"StringValue": filters["ticker"]}
			}
		})
	if isinstance(filters.get("fiscal_year"), int):
		and_filters.append({
			"EqualsTo": {
				"Key": "fiscal_year",
				"Value": {"LongValue": filters["fiscal_year"]}
			}
		})
	if not and_filters:
		return {}
	return {"AndAllFilters": and_filters}


def query_kendra_snippets(themes: List[str], max_per_theme: int = 5, *, attribute_filters: Dict[str, Any] | None = None) -> Dict[str, List[Dict[str, Any]]]:
	results: Dict[str, List[Dict[str, Any]]] = {}
	for theme in themes:
		try:
			# Bias queries to 10-K content
			query_text = f'"{theme}" AND ("10-K" OR "Form 10-K")'
			kwargs = {"IndexId": KENDRA_INDEX_ID, "QueryText": query_text, "PageSize": 10}
			attr_filter = _build_attribute_filter(attribute_filters or {})
			if attr_filter:
				kwargs["AttributeFilter"] = attr_filter
			resp = kendra.query(**kwargs)
		except Exception as e:
			results[theme] = [{"error": f"Kendra query failed: {e}"}]
			continue

		snippets: List[Dict[str, Any]] = []
		for item in resp.get("ResultItems", [])[: max_per_theme * 2]:  # fetch a bit extra to filter
			etype = item.get("Type")
			# Prefer ANSWER/DOCUMENT excerpts
			excerpt = item.get("DocumentExcerpt", {}).get("Text", "")
			title = item.get("DocumentTitle", {}).get("Text", "")
			url = item.get("DocumentURI") or item.get("DocumentId")
			score = item.get("ScoreAttributes", {}).get("ScoreConfidence")
			if excerpt:
				snippets.append({
					"title": title,
					"url": url,
					"excerpt": excerpt[:1500],
					"type": etype,
					"score": score,
				})
			if len(snippets) >= max_per_theme:
				break

		results[theme] = snippets
	return results


# ---- Optional: Load market data from S3 and merge ----
def _read_s3_csv_to_io(bucket: str, key: str) -> io.StringIO:
	obj = s3.get_object(Bucket=bucket, Key=key)
	data = obj["Body"].read().decode("utf-8", errors="ignore")
	return io.StringIO(data)

def attach_market_data_if_available(event: Dict[str, Any]) -> Dict[str, Any]:
	"""Optionally load S&P500 composition and performance CSVs from S3 and merge.
	Expects event keys: sp500_composition_key, stocks_performance_key.
	Returns a small summary to keep Lambda responses light.
	"""
	comp_key = (event or {}).get("sp500_composition_key")
	perf_key = (event or {}).get("stocks_performance_key")
	if not comp_key or not perf_key:
		return {"attached": False, "reason": "missing_keys"}

	try:
		# Import on-demand so Lambda can run even without pandas packaged
		import pandas as pd  # type: ignore
		# Import project loader
		from data_loader import load_sp500_composition, load_stocks_performance, merge_sp500_data  # type: ignore

		comp_io = _read_s3_csv_to_io(S3_BUCKET, comp_key)
		perf_io = _read_s3_csv_to_io(S3_BUCKET, perf_key)

		df_comp = load_sp500_composition(comp_io)
		df_perf = load_stocks_performance(perf_io)
		df_merged = merge_sp500_data(df_comp, df_perf)
		if df_merged is None or df_merged.empty:
			return {"attached": False, "reason": "merge_empty"}

		# Lightweight summary (avoid returning full dataframe)
		sample = df_merged.head(10).to_dict(orient="records")
		total_cap = float(df_merged["Market Cap"].sum()) if "Market Cap" in df_merged.columns else 0.0
		return {
			"attached": True,
			"count": int(len(df_merged)),
			"sample": sample,
			"total_market_cap": total_cap,
		}
	except Exception as e:
		return {"attached": False, "reason": f"{e}"}


# ---- Step 3: Synthesize final report with Claude Haiku ----
def synthesize_report(themes: List[str], kendra_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
	system = (
		"You are a senior equity analyst specializing in regulatory impacts. "
		"Return ONLY strict JSON."
	)
	payload = {
		"themes": themes,
		"kendra_results": kendra_results,
	}
	user = (
		"Using the finance risk themes and Kendra 10-K snippets, produce a FINAL STRICT JSON report with the TOP 5 most exposed public companies.\n"
		"Schema EXACTLY: {\n"
		"  \"companies\": [\n"
		"    {\"company\": \"string\", \"exposure_reason\": \"<=2 sentences, financial impact & why\"}\n"
		"  ]\n"
		"}\n\n"
		"Constraints:\n"
		"- Use ONLY evidence from the provided snippets to name companies.\n"
		"- Prefer companies explicitly named in excerpts; avoid speculation.\n"
		"- Keep reasons finance-focused (costs, margins, revenue mix, geography, supply chain).\n"
		"- Be concise and avoid duplications.\n\n"
		f"Data:\n{json.dumps(payload)[:12000]}"
	)
	result = _invoke_bedrock_json(MODEL_HAIKU, user, system_text=system, max_tokens=800, temperature=0.2)
	# Ensure shape
	companies = result.get("companies", [])
	if not isinstance(companies, list):
		companies = []
	# Truncate to top 5
	result["companies"] = companies[:5]
	return result


# ---- Lambda Handler ----
def lambda_handler(event, context):
	"""AWS Lambda entrypoint.

	Event expected to contain:
	{"regulation_text": "..."}
	Optionally: {"kendra_index_id": "..."}
	"""
	try:
		regulation_text = (event or {}).get("regulation_text", "")
		if not regulation_text:
			return {"statusCode": 400, "body": json.dumps({"error": "Missing regulation_text"})}

		# Allow overriding Kendra index via event
		global KENDRA_INDEX_ID
		if "kendra_index_id" in (event or {}):
			KENDRA_INDEX_ID = event["kendra_index_id"]

		# Step 1: Extract themes
		themes = extract_risk_themes(regulation_text)
		if not themes:
			# Minimal fallback: heuristics
			themes = [w for w in ["tax", "tariffs", "supply chain"] if w in regulation_text.lower()][:3] or ["regulatory risk"]

		# Step 2: Kendra search with optional attribute filters
		attribute_filters = (event or {}).get("kendra_filters", {})
		kendra_results = query_kendra_snippets(themes, max_per_theme=5, attribute_filters=attribute_filters)

		# Optional: attach market data from S3 if provided
		market_data = attach_market_data_if_available(event or {})

		# Step 3: Synthesis
		final_report = synthesize_report(themes, kendra_results)
		# Append market snapshot if available (non-blocking)
		if market_data.get("attached"):
			final_report["market_data"] = {
				"count": market_data.get("count"),
				"total_market_cap": market_data.get("total_market_cap"),
				"sample": market_data.get("sample"),
			}

		# Return final JSON
		return {
			"statusCode": 200,
			"body": json.dumps(final_report)
		}

	except Exception as e:
		return {
			"statusCode": 500,
			"body": json.dumps({"error": str(e)})
		}

