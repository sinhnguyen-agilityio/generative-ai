# HTTP 429 “Too Many Requests”: RFC definition, semantics, and retry-delay header

## Scope
This report identifies the RFC and section that define HTTP status code 429 “Too Many Requests,” and summarizes what the provided RFC excerpts say about the code’s meaning and retry delay via a specific header. It also states what the provided excerpts do not explicitly cover regarding broader client retry behavior.

## RFC that defines HTTP 429 and the section that defines it
The provided RFC excerpt identifies **RFC 6585 (Additional HTTP Status Codes)** as the document containing the **“4. 429 Too Many Requests”** section. [S13][S14]

## Precise semantics/meaning of HTTP 429 and RFC guidance relevant to client retry behavior
The provided RFC excerpt states that **the 429 status code indicates that the user has sent too many requests in a given amount of time (“rate limiting”)**. [S13]

For that condition, the excerpt states that **the response representations SHOULD include details explaining the condition** and **MAY include a `Retry-After` header indicating how long to wait before making a new request**. [S13]

## HTTP response header that indicates how long to wait before retrying (name and RFC meaning)
The RFC excerpt specifies that the response representations **MAY include a `Retry-After` header indicating how long to wait before making a new request**. [S13]

## Limitations
- The provided evidence does not include the RFC Editor text for the relevant section; instead, it provides an httpwg.org RFC 6585 rendering/mirror excerpt. [S13]
- The provided RFC excerpt does not include any additional explicit client retry-behavior rules beyond the `Retry-After` header’s meaning (for example, it does not provide explicit client `SHOULD`/`MUST`/`SHOULD-NOT` retry timing rules other than the header’s “how long to wait” indication). [S13]
- The provided evidence does not include any RFC-specified constraints about `Retry-After` formatting or interpretation (e.g., seconds vs HTTP-date), because such details are not present in the supplied excerpt. [S13]

## Sources
- [S13] RFC 6585 - Additional HTTP Status Codes - https://httpwg.org/specs/rfc6585.html
- [S14] RFC 6585: Additional HTTP Status Codes | RFC Editor - https://www.rfc-editor.org/info/rfc6585