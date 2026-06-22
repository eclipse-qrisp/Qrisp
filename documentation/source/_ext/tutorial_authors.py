"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

from docutils import nodes

LINKEDIN_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="#0077b5" '
    'viewBox="0 0 24 24" style="vertical-align: middle; margin-left: 4px; position: relative; top: -1px;">'
    '<path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>'
    "</svg>"
)

tutorial_authors = {
    "general/tutorial/BE_tutorial/BE_vol1": [
        {
            "name": "Matic Petrič",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
        {
            "name": "René Zander",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/BE_tutorial/BE_vol2": [
        {
            "name": "Matic Petrič",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
        {
            "name": "René Zander",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/BigInteger": [
        {
            "name": "Eric Kühnke",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
        {
            "name": "Raphael Seidel",
            "affiliation": "IQM Quantum Computers",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/CD": [
        {
            "name": "Carlotta Koroll",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
        {
            "name": "Sebastian Bock",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/FT_compilation": [
        {
            "name": "Raphael Seidel",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/Jasp": [
        {
            "name": "Raphael Seidel",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/JaspQAOAtutorial": [
        {
            "name": "René Zander",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/GQSP_filtering": [
        {
            "name": "René Zander",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/HHL": [
        {
            "name": "René Zander",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
        {
            "name": "Raphael Seidel",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
        {
            "name": "Matic Petrič",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/H2": [
        {
            "name": "Raphael Seidel",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/QAOAtutorial/ConstrainedMixers": [
        {
            "name": "Raphael Seidel",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/QAOAtutorial/MaxCut": [
        {
            "name": "Matic Petrič",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/QAOAtutorial/MkCS": [
        {
            "name": "Matic Petrič",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/QAOAtutorial/PortfolioRebalancing": [
        {
            "name": "Niklas Steinmann",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/QAOAtutorial/QUBO": [
        {
            "name": "Matic Petrič",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
        {
            "name": "René Zander",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/QIROtutorial": [
        {
            "name": "Niklas Steinmann",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/QMCItutorial": [
        {
            "name": "Niklas Steinmann",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
        {
            "name": "René Zander",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/Shor": [
        {
            "name": "Raphael Seidel",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/Sudoku": [
        {
            "name": "Raphael Seidel",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
        {
            "name": "René Zander",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
        {
            "name": "Matic Petrič",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
        {
            "name": "Niklas Steinmann",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/TSP": [
        {
            "name": "Raphael Seidel",
            "affiliation": "Fraunhofer FOKUS",
            "role": "Eclipse Qrisp Contributor",
            "linkedin": None,
        },
    ],
    "general/tutorial/tutorial": [
        {
            "name": "The Qrisp authors",
            "affiliation": "Eclipse Qrisp Contributors",
            "role": "Eclipse Qrisp Contributors",
            "linkedin": None,
        },
    ],
}


def inject_authors_after_title(app, doctree):
    # Get the current document name being processed
    docname = app.builder.env.docname

    # If the document isn't in our dictionary, skip it
    if docname not in tutorial_authors:
        return

    # --- Step A: Build the HTML Content ---
    author_list = tutorial_authors[docname]
    affiliations_dict = {}

    # Group authors by affiliation
    for author in author_list:
        affiliation = author.get("affiliation", "Unaffiliated")
        name = author["name"]
        linkedin_url = author.get("linkedin")

        # Build the author's name string (with optional LinkedIn icon)
        author_html = f"<span style='font-weight: 600; color: #222;'>{name}</span>"
        if linkedin_url:
            author_html += f'<a href="{linkedin_url}" target="_blank" title="{name} on LinkedIn" style="text-decoration: none;">{LINKEDIN_ICON}</a>'

        if affiliation not in affiliations_dict:
            affiliations_dict[affiliation] = []
        affiliations_dict[affiliation].append(author_html)

    # Create the flattened sentence structure
    affiliation_blocks = []
    for affiliation, author_html_list in affiliations_dict.items():
        names_joined = ", ".join(author_html_list)
        block = f"{names_joined}, <span style='color: #666; font-style: italic;'>{affiliation}</span>"
        affiliation_blocks.append(block)

    final_text = "; ".join(affiliation_blocks) + "."

    # Wrap in the final styled container
    html_content = f"""
    <div class="author-bio" style="margin-bottom: 24px; padding: 15px; background: #fdfdfd; border: 1px solid #eee; border-left: 5px solid #007acc; border-radius: 4px; font-family: sans-serif; line-height: 1.6;">
        <strong style="color: #333; margin-right: 5px;">Authors:</strong> 
        {final_text}
    </div>
    """

    # Convert HTML string to a Docutils node
    bio_node = nodes.raw("", html_content, format="html")

    # --- Step B: Inject into the AST ---
    # Handle backwards compatibility for older Sphinx versions
    finder = doctree.findall if hasattr(doctree, "findall") else doctree.traverse

    injected = False

    # Hunt for the main document title
    for title_node in finder(nodes.title):
        # Security Check: Ensure this title belongs to the main page layout,
        # not a warning box, note, or sidebar.
        if isinstance(title_node.parent, nodes.section):
            parent = title_node.parent
            index = parent.index(title_node)

            # Insert our HTML block immediately AFTER the title
            parent.insert(index + 1, bio_node)
            injected = True

            # Stop searching after finding the first valid H1
            break

    # Fallback: If the author forgot to include a title cell entirely,
    # shove the bio to the absolute top of the page so it isn't lost.
    if not injected:
        doctree.insert(0, bio_node)


# Register the Hook with Sphinx
def setup(app):
    # 'doctree-read' runs immediately after the notebook/markdown is parsed into AST
    app.connect("doctree-read", inject_authors_after_title)

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
