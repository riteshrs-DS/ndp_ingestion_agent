# src/ckan_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests


@dataclass(frozen=True)
class CkanClient:
    """
    CKAN Action API client.
    Docs: /api/3/action/<action_name>  (package_search, package_show, organization_show, etc.)
    """
    base_url: str  # e.g. "https://ndp-test.sdsc.edu/catalog"

    def _action_url(self, action: str) -> str:
        return f"{self.base_url.rstrip('/')}/api/3/action/{action}"

    def action_get(self, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self._action_url(action)
        r = requests.get(url, params=params or {}, timeout=60)
        r.raise_for_status()
        payload = r.json()
        if not payload.get("success", False):
            raise RuntimeError(f"CKAN action failed: {payload}")
        return payload["result"]

    def organization_show(self, org: str) -> Dict[str, Any]:
        return self.action_get("organization_show", {"id": org})

    def package_search(
        self,
        fq: str,
        rows: int = 10,
        start: int = 0,
        sort: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"fq": fq, "rows": rows, "start": start}
        if sort:
            params["sort"] = sort
        return self.action_get("package_search", params)

    def list_org_packages(self, org: str, rows: int = 20, start: int = 0) -> Dict[str, Any]:
        # CKAN package_search filtered by organization
        return self.package_search(fq=f"organization:{org}", rows=rows, start=start)

    def get_package(self, dataset_id_or_name: str) -> Dict[str, Any]:
        return self.package_show(dataset_id_or_name)

    def get_package_show(self, dataset_id_or_name: str) -> Dict[str, Any]:
        return self.package_show(dataset_id_or_name)

    #  def package_show(self, dataset_id_or_name: str) -> Dict[str, Any]:
  #      return self.action_get("package_show", {"id": dataset_id_or_name})

    def download(self, url: str) -> bytes:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return r.content

    def package_show(self, dataset_id_or_name: str) -> Dict[str, Any]:
        return self.action_get("package_show", {"id": dataset_id_or_name})


def build_fq_for_org(org: str, res_format: Optional[str] = None) -> str:
    """
    CKAN uses Solr filter queries in 'fq'.
    Examples: organization:bco_weather + res_format:XML
    """
    parts = [f"organization:{org}"]
    if res_format:
        # CKAN facets use res_format (as seen in org page facets)
        parts.append(f"res_format:{res_format}")
    return " ".join(parts)


def iter_all_packages(
    client: CkanClient,
    org: str,
    res_format: Optional[str] = None,
    page_size: int = 100,
    max_total: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch all packages for an organization (optionally filtered by res_format).
    Returns list of package 'results' stubs from package_search.
    """
    fq = build_fq_for_org(org, res_format=res_format)
    start = 0
    out: List[Dict[str, Any]] = []

    while True:
        batch = client.package_search(fq=fq, rows=page_size, start=start)
        results = batch.get("results", [])
        out.extend(results)

        if max_total is not None and len(out) >= max_total:
            return out[:max_total]

        count = batch.get("count", 0)
        start += page_size
        if start >= count or not results:
            break

    return out
