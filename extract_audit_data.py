"""Extract AuditData JSON from the Copilot interactions export.

Reads input/Copilot-Interactions-Model.csv, writes:
  output/audit_data.json  — one JSON array of the parsed AuditData objects
  output/audit_data.csv   — one readable row per interaction
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "input" / "Copilot-Interactions-Model.csv"
OUT_DIR = ROOT / "output"
JSON_PATH = OUT_DIR / "audit_data.json"
CSV_PATH = OUT_DIR / "audit_data.csv"

COLUMNS = [
    "Id",
    "CreationTime",
    "UserId",
    "UserKey",
    "ClientIP",
    "ClientRegion",
    "AppHost",
    "AppIdentity",
    "AgentName",
    "AgentId",
    "ApplicationName",
    "LicenseType",
    "ConversationId",
    "ThreadId",
    "MessageCount",
    "PromptCount",
    "ResponseCount",
    "Models",
    "ModelProviders",
    "Plugins",
    "ContextTypes",
    "ContextCount",
    "ResourceCount",
    "ResourceTypes",
    "ResourceNames",
    "ResourceUrls",
    "MemoryUpdated",
    "DLPEvaluationDeferred",
    "SensitivityLabelId",
]


def join_unique(values):
    seen = []
    for value in values:
        text = "" if value is None else str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return "; ".join(seen)


def clean_row(record):
    event = record.get("CopilotEventData") or {}
    messages = event.get("Messages") or []
    models = event.get("ModelTransparencyDetails") or []
    plugins = event.get("AISystemPlugin") or []
    contexts = event.get("Contexts") or []
    resources = event.get("AccessedResources") or []

    prompts = sum(1 for message in messages if message.get("isPrompt") is True)
    responses = sum(1 for message in messages if message.get("isPrompt") is False)

    return {
        "Id": record.get("Id", ""),
        "CreationTime": record.get("CreationTime", ""),
        "UserId": record.get("UserId", ""),
        "UserKey": record.get("UserKey", ""),
        "ClientIP": record.get("ClientIP", ""),
        "ClientRegion": record.get("ClientRegion", ""),
        "AppHost": event.get("AppHost", ""),
        "AppIdentity": record.get("AppIdentity", ""),
        "AgentName": record.get("AgentName") or event.get("TargetAgentName") or "",
        "AgentId": record.get("AgentId") or event.get("TargetPlatformAgentId") or "",
        "ApplicationName": record.get("ApplicationName", ""),
        "LicenseType": event.get("LicenseType", ""),
        "ConversationId": event.get("ConversationId", ""),
        "ThreadId": event.get("ThreadId", ""),
        "MessageCount": len(messages),
        "PromptCount": prompts,
        "ResponseCount": responses,
        "Models": join_unique(item.get("ModelName") for item in models),
        "ModelProviders": join_unique(item.get("ModelProviderName") for item in models),
        "Plugins": join_unique(item.get("Name") or item.get("Id") for item in plugins),
        "ContextTypes": join_unique(item.get("Type") for item in contexts),
        "ContextCount": len(contexts),
        "ResourceCount": len(resources),
        "ResourceTypes": join_unique(item.get("Type") for item in resources),
        "ResourceNames": join_unique(item.get("Name") for item in resources),
        "ResourceUrls": join_unique(item.get("SiteUrl") for item in resources),
        "MemoryUpdated": event.get("MemoryUpdated", ""),
        "DLPEvaluationDeferred": event.get("DLPEvaluationDeferred", ""),
        "SensitivityLabelId": event.get("SensitivityLabelId", ""),
    }


def load_records():
    csv.field_size_limit(10**8)
    records = []
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "AuditData" not in (reader.fieldnames or []):
            raise SystemExit(f"AuditData column not found in {SOURCE}")
        for row in reader:
            raw = (row.get("AuditData") or "").strip()
            if raw:
                records.append(json.loads(raw))
    return records


def main():
    records = load_records()
    OUT_DIR.mkdir(exist_ok=True)

    with JSON_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(clean_row(record))

    print(f"records: {len(records)}")
    print(f"columns: {len(COLUMNS)}")
    print(f"json: {JSON_PATH}")
    print(f"csv:  {CSV_PATH}")


if __name__ == "__main__":
    main()
