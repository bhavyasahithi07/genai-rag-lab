from langchain_text_splitters import HTMLHeaderTextSplitter

#instead we can use a url too
HTML_String="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample HTML</title>
</head>
<body>

    <h1>Welcome to My Website</h1>

    <p>
        This is a sample HTML page for testing. It contains headings,
        paragraphs, lists, links, and a table.
    </p>

    <h2>My Favorite Programming Languages</h2>
    <ul>
        <li>Python</li>
        <li>Java</li>
        <li>JavaScript</li>
    </ul>

    <h2>Useful Link</h2>
    <a href="https://www.python.org">Visit Python Official Website</a>

    <h2>Student Details</h2>
    <table border="1">
        <tr>
            <th>Name</th>
            <th>Course</th>
            <th>Year</th>
        </tr>
        <tr>
            <td>Bhavya</td>
            <td>Computer Science</td>
            <td>2026</td>
        </tr>
        <tr>
            <td>John</td>
            <td>Data Science</td>
            <td>2025</td>
        </tr>
    </table>

    <h2>Contact</h2>
    <form>
        <label>Name:</label><br>
        <input type="text" placeholder="Enter your name"><br><br>

        <label>Email:</label><br>
        <input type="email" placeholder="Enter your email"><br><br>

        <input type="submit" value="Submit">
    </form>

</body>
</html>
"""

header_splitting=[
    ("h1","header1"),
    ("h2","header2")
]

html_splitter_char=HTMLHeaderTextSplitter(headers_to_split_on=header_splitting)
html_header_splits=html_splitter_char.split_text(HTML_String)#use that url in place of HTML_String
print(html_header_splits[0])