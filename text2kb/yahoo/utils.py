import codecs
import json
from sys import argv


def convert_serp_file(serp_path, out_path):
    with codecs.open(serp_path, 'r', encoding='utf8') as inp:
        serps = json.load(inp)

    res_serps = []
    res = {"elems": res_serps}
    for serp in serps:
        search_results = []
        res_serp = {"question": serp["query"], "results": {"elems": search_results}}

        for r in serp["searchResults"]:
            search_results.append(r)
        res_serps.append(res_serp)

    with codecs.open(out_path, 'w', encoding='utf8') as out:
        json.dump(res, out, indent=2)


def create_dummy_documents_file(serp_path, pages_index, out_path):
    with codecs.open(serp_path, 'r', encoding='utf8') as inp:
        serps = json.load(inp)

    page_files = {}
    with open(pages_index, 'r') as inp:
        for line in inp:
            url, path = line.strip().split("\t")
            page_files[url] = path

    res_docs = []
    res = {"elems": res_docs}
    for serp in serps:
        for r in serp["searchResults"]:
            r["query"] = serp["query"]
            r["contentDocument"] = page_files[r["url"]] if r["url"] in page_files else []
            res_docs.append(r)

    with codecs.open(out_path, 'w', encoding='utf8') as out:
        json.dump(res, out, indent=2)


def extract_qids(serp_file, out_file):
    with codecs.open(serp_file, 'r', encoding='utf8') as inp:
        serps = json.load(inp)
    res = []
    for serp in serps:
        query = ' '.join(serp["query"].split('"')[:-1]).strip()
        for r in serp["searchResults"]:
            title = r["title"].split(" | ")
            title = title[0].strip(". ")
            if title.startswith(query) or query.startswith(title):
                qid = r["url"].split("qid=")[1]
                res.append({"question": query, "qid": qid})
            else:
                print(query)
                print(r)
                print("-----")

    with codecs.open(out_file, 'w', encoding="utf8") as out:
        json.dump(res, out, indent=2)




if __name__ == "__main__":
    convert_serp_file(argv[1], argv[2])
    create_dummy_documents_file(argv[1], argv[3], argv[4])

    # extract_qids(argv[1], argv[2])