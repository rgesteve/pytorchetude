---
_ns:
    xsd: http://www.w3.org/2001/XMLSchema#
    foaf: http://xmlns.com/foaf/0.1/
_id: http://me.markus-lanthaler.com/
a: foaf_Person
foaf_name: Markus Lanthaler
foaf_homepage: http://www.markus-lanthaler.com/
foaf_depiction: http://twitter.com/account/profile_image/markuslanthaler
---

Using the aREF serialization of RDF a blockquote and footnote[^1]:

> This is what someone said.

---

This is a [link to this post]({{ ref "2018/09/first-post" }}) and [another link]({{ ref "2018/09/first-post" }}).

---

[^1]: This would be a footnote with a [link to first post]({{ ref "2018/09/first-post" }}).
