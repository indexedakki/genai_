import fitz  # PyMuPDF

def extract_section_from_pdf(pdf_path, section_title):
    # Open the PDF file
    document = fitz.open(pdf_path)
    
    # Initialize variables
    section_text = ""
    section_found = False
    
    # Iterate through each page
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text = page.get_text("text")
        
        # Split the text into lines
        lines = text.split('\n')
        
        for line in lines:
            # Check if the line contains the section title
            if section_title.lower() in line.lower():
                section_found = True
            
            # If the section is found, start collecting text
            if section_found:
                section_text += line + "\n"
                
                # Stop collecting text if the next section title is found
                if line.strip().endswith(":"):
                    section_found = False
                    break
    
    return section_text

# Example usage
pdf_path = "000531869.pdf"
section_title = "Results"  # The title of the section you want to extract
section_content = extract_section_from_pdf(pdf_path, section_title)

print("Extracted Section Content:")
print(section_content)