import json
import re

INPUT_FILE = r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\Data\updated_knowledge_base.json"
OUTPUT_FILE = r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\Data\cleaned_updated_knowledge_base.json"


def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_navigation_junk(text):
    junk_patterns = [
        r'Back',
        r'Forward',
        r'Alle Elemente ausklappen',
        r'Alle Elemente einklappen',
        r'Inhalt ausklappen',
        r'Inhalt einklappen',
        r'Discover',
        r'More on the topic',
        r'See also',
        r'Show Video',
        r'Links to the topic',
        r'Continue',
        r'Foto:.*?',
        r'Video:.*?',
    ]

    for pattern in junk_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text


def clean_title(title):
    # Removing noise
    title = title.split('\n')[0]
    return title.strip()


def classify_page(url):

    #  DEADLINES 
    if "master-application-deadlines" in url:
        return "deadlines"
    elif "degree-programs/datasciencems" in url:
        return "program_overview"

    # SPECIFIC APPLICATION TYPES 
    elif "assist" in url:
        return "application_foreign"

    elif "marvin" in url:
        return "application_german"

    elif "eap" in url:
        return "eligibility_assessment"

    #  PROGRAM STRUCTURE 
    elif "study-structure" in url:
        return "study_structure"

    elif "examination-regulations" in url:
        return "exam_regulations"

    elif "study-requirements" in url:
        return "admission_core"

    elif "modules-and-courses" in url:
        return "modules"

    elif "glossary" in url:
        return "modules_glossary"

    elif "types-of-courses" in url:
        return "types_of_module"

    #  STUDENT LIFE 
    elif "visa" in url:
        return "visa"

    elif "finance" in url:
        return "finance"

    elif "contributions-and-fees" in url:
        return "fees"

    elif "explore-marburg" in url:
        return "city_life"

    #  MARKETING 
    elif "therefore" in url or "perspectives" in url:
        return "marketing"

    elif "contact" in url:
        return "contact"

    # GENERIC APPLICATION 
    elif "application" in url:
        return "application"

    # REST OF THE PAGES 
    else:
        return "general"


'''def classify_page(url):
    if "master-application-deadlines" in url:
        return "deadlines"
    elif "application" in url:
        return "application"
    elif "assist" in url:
        return "application_foreign"
    elif "marvin" in url:
        return "application_german"
    elif "eap" in url:
        return "eligibility_assessment"
    elif "visa" in url:
        return "visa"
    elif "finance" in url:
        return "finance-and-scholarships"
    elif "contributions-and-fees" in url:
        return "fees"
    elif "study-structure" in url:
        return "study_structure"
    elif "modules-and-courses" in url:
        return "modules"
    elif "study-requirements" in url:
        return "admission_core"
    elif "examination-regulations" in url:
        return "exam_regulations"
    elif "glossary" in url:
        return "glossary"
    elif "types-of-courses" in url:
        return "types_of_module"
    elif "therefore" in url or "perspectives" in url:
        return "marketing"
    elif "contact" in url:
        return "contact"
    elif "explore-marburg" in url:
        return "city_life"
    else:
        return "general"'''


import re

def extract_deadlines_structured(content, url):
    deadlines_data = {
        "url": url,
        "type": "deadlines",
        "program": "Data Science (M.Sc.)",
        "deadlines": []
    }

    date_pattern = r"[A-Za-z]+\s\d{1,2}(?:st|nd|rd|th),\s\d{4}"

    pattern = (
        r"Data Science \(M\.Sc\.\):.*?"
        r"foreign university degree:\s*(" + date_pattern + ").*?"
        r"German university degree:\s*(" + date_pattern + ").*?"
        r"end of the deadline:\s*(" + date_pattern + ")"
    )

    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        foreign_start = match.group(1)
        german_start = match.group(2)
        deadline = match.group(3)

        # Determine semester from month
        month = foreign_start.split()[0]

        if month in ["November", "December", "January", "February"]:
            semester = "Summer"
        else:
            semester = "Winter"

        deadlines_data["deadlines"].append({
            "semester": semester,
            "foreign_degree": {
                "start": foreign_start,
                "deadline": deadline
            },
            "german_degree": {
                "start": german_start,
                "deadline": deadline
            }
        })

    return deadlines_data

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen_urls = set()
    cleaned_data = []

    for entry in data:
        url = entry["url"]

        # Removing duplicate URLs
        if url in seen_urls:
            continue
        seen_urls.add(url)

        title = clean_title(entry["title"])
        content = entry["content"]

        content = remove_navigation_junk(content)
        content = normalize_whitespace(content)

        page_type = classify_page(url)

        if page_type == "deadlines":
            cleaned_entry = extract_deadlines_structured(content, url)
        else:
            cleaned_entry = {
                "url": url,
                "title": title,
                "type": page_type,
            "clean_content": content
        }

        cleaned_data.append(cleaned_entry)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print("Cleaning complete. Output saved.")


if __name__ == "__main__":
    main()