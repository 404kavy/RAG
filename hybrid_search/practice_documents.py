from langchain_core.documents import Document

# Practice dataset for testing hybrid retrieval (BM25 + vector search).

documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports gigabit speeds, dual-band Wi-Fi, and advanced QoS features.",
        metadata={"type": "product", "sku": "SKU-7742X"},
    ),
    Document(
        page_content="Product SKU-8810A is a compact network switch with eight Ethernet ports and VLAN support.",
        metadata={"type": "product", "sku": "SKU-8810A"},
    ),
    Document(
        page_content="The AeroMesh X5 is a mesh Wi-Fi system designed to improve wireless coverage in large homes and offices.",
        metadata={"type": "product", "sku": "AEROMESH-X5"},
    ),
    Document(
        page_content="For network connectivity issues, first check the Ethernet cable, router power, and status indicator lights.",
        metadata={"type": "troubleshooting", "topic": "connectivity"},
    ),
    Document(
        page_content="If Wi-Fi coverage is weak, reposition the access point, reduce physical obstructions, and check for channel interference.",
        metadata={"type": "troubleshooting", "topic": "wifi"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates that a remote host rejected a connection attempt. Check whether the service is running and whether firewall rules allow the connection.",
        metadata={"type": "error", "error_code": "E_CONN_REFUSED"},
    ),
    Document(
        page_content="Error code E_DNS_LOOKUP means the hostname could not be resolved. Verify the DNS server configuration and the hostname spelling.",
        metadata={"type": "error", "error_code": "E_DNS_LOOKUP"},
    ),
    Document(
        page_content="Error code AUTH-401 indicates that the request was not authenticated. Check the access token and its expiration time.",
        metadata={"type": "error", "error_code": "AUTH-401"},
    ),
    Document(
        page_content="The authentication process requires valid credentials. Use OAuth 2.0 authorization flows for secure API access, and never store client secrets in source code.",
        metadata={"type": "authentication", "protocol": "OAuth 2.0"},
    ),
    Document(
        page_content="WCAG stands for Web Content Accessibility Guidelines. It provides recommendations for making web content accessible to people with disabilities.",
        metadata={"type": "acronym", "term": "WCAG"},
    ),
    Document(
        page_content="In networking, QoS means Quality of Service. QoS policies can prioritize latency-sensitive traffic such as voice and video.",
        metadata={"type": "acronym", "term": "QoS"},
    ),
    Document(
        page_content="John Smith is the account manager for the Northwind deployment. Contact the account team for contract and billing questions.",
        metadata={"type": "person", "name": "John Smith", "department": "accounts"},
    ),
    Document(
        page_content="John Smith's Bakery is a small business using the AeroMesh X5 wireless network for its point-of-sale terminals.",
        metadata={"type": "business", "name": "John Smith's Bakery"},
    ),
    Document(
        page_content="The accounting policy explains invoice approval, expense categorization, reconciliation, and monthly financial reporting.",
        metadata={"type": "policy", "topic": "accounting"},
    ),
    Document(
        page_content="To update router firmware, download the firmware package for the exact hardware revision, back up the configuration, and follow the vendor's installation procedure.",
        metadata={"type": "procedure", "topic": "firmware"},
    ),
    Document(
        page_content="A VLAN is a virtual local area network. It logically separates devices into network segments without requiring separate physical switches.",
        metadata={"type": "technical_concept", "term": "VLAN"},
    ),
    Document(
        page_content="A firewall controls incoming and outgoing network traffic using a set of security rules. Incorrect rules can block legitimate connections.",
        metadata={"type": "technical_concept", "term": "firewall"},
    ),
    Document(
        page_content="The service-level objective for the support API is 99.9 percent availability per calendar month, excluding scheduled maintenance.",
        metadata={"type": "service_level", "service": "support_api"},
    ),
    Document(
        page_content="When an API returns HTTP 429, the client has sent too many requests. Apply exponential backoff and respect the Retry-After header when it is present.",
        metadata={"type": "troubleshooting", "topic": "rate_limiting"},
    ),
    Document(
        page_content="The Northwind office network uses router SKU-7742X, switch SKU-8810A, and a separate guest Wi-Fi network for visitors.",
        metadata={"type": "deployment", "customer": "Northwind"},
    ),
]

if __name__ == "__main__":
    print(f"Loaded {len(documents)} practice documents.")
    for index, document in enumerate(documents, start=1):
        print(f"\nDocument {index}: {document.metadata}")
        print(document.page_content)
