import os
import time
import xml.etree.ElementTree as ET
from Bio import Entrez
import requests
import urllib.request

# NCBI requires you to set your email and tool name
Entrez.email = "your.email@example.com"  # Replace with your email
Entrez.tool = "PubMedDownloader/1.0"

def search_and_download_pmc_articles(drug_name, max_results=10, output_dir="articles"):
    """
    Search PMC for free full-text articles related to a drug and download PDFs/XML
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Search PMC for the drug name
        handle = Entrez.esearch(
            db="pmc",
            term=f"{drug_name} clinical study",
            retmax=max_results,
            sort="relevance"
        )
        search_results = Entrez.read(handle)
        pmc_ids = search_results["IdList"]
        print(f"Found {len(pmc_ids)} articles for {drug_name}")

        # Step 2: Download articles
        for pmc_id in pmc_ids:
            try:
                # Get download links from PMC OA service
                oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmc_id}"
                response = requests.get(oa_url)
                root = ET.fromstring(response.content)

                # Find all PDF download links
                pdf_urls = []
                for link in root.findall(".//link"):
                    if link.get("format") == "pdf":
                        pdf_urls.append(link.get("href"))

                # Select the first HTTPS link, or fallback to any URL
                pdf_url = None
                for url in pdf_urls:
                    if url.startswith('http'):
                        pdf_url = url
                        break
                if not pdf_url and pdf_urls:
                    pdf_url = pdf_urls[0]  # Could be FTP

                # Download PDF if available
                if pdf_url:
                    print(f"Downloading PMC{pmc_id} from {pdf_url}...")
                    try:
                        if pdf_url.startswith('ftp'):
                            # Use urllib to handle FTP
                            urllib.request.urlretrieve(
                                pdf_url,
                                os.path.join(output_dir, f"PMC{pmc_id}.pdf")
                            )
                        else:
                            # Use requests for HTTP/HTTPS
                            pdf_response = requests.get(pdf_url)
                            pdf_response.raise_for_status()
                            with open(os.path.join(output_dir, f"PMC{pmc_id}.pdf"), "wb") as f:
                                f.write(pdf_response.content)
                        print(f"Successfully downloaded PMC{pmc_id}")
                    except Exception as e:
                        print(f"Error downloading PMC{pmc_id}: {str(e)}")
                else:
                    print(f"No PDF available for PMC{pmc_id}")

                # Respect NCBI's rate limit (3 requests/sec)
                time.sleep(0.34)

            except Exception as e:
                print(f"Error processing PMC{pmc_id}: {str(e)}")

    except Exception as e:
        print(f"Search failed: {str(e)}")

if __name__ == "__main__":
    # Example usage
    search_and_download_pmc_articles(
        drug_name="Sevelamer",
        max_results=20,
        output_dir="clinical_studies"
    )