def format_source_label(source: dict) -> str:
    filename = source.get("source", "Unknown source")
    page = source.get("page")
    if page is not None:
        return f"{filename} — Page {page}"
    return f"{filename} — Page reference unavailable"


def get_unique_sources(documents) -> list[str]:
    return sorted(
        {
            doc.metadata.get("source", "Unknown source")
            for doc in documents
        }
    )
