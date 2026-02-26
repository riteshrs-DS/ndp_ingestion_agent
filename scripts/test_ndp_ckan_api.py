from src.ckan_client import CkanClient, iter_all_packages

BASE = "https://ndp-test.sdsc.edu/catalog"
client = CkanClient(BASE)

for org, fmt in [
    ("bco_weather", "XML"),
    ("clm_test", "TXT"),
    ("wfsi", "EML"),   # try; EML is often an XML format label in CKAN
]:
    pkgs = iter_all_packages(client, org=org, res_format=fmt, page_size=50, max_total=5)
    print(f"\nORG={org} FORMAT={fmt} sample_count={len(pkgs)}")
    for p in pkgs:
        print(" -", p.get("name"), "|", p.get("title"))
