# HTTP 401 vs 403: Evidence-Grounded Constraints and Incomplete RFC Semantics

## Scope
This report attempts to explain the normative meanings and header requirements for HTTP status codes 401 (Unauthorized) and 403 (Forbidden), using only the supplied research findings and source excerpts. As of 2026-10-04, the provided evidence is insufficient to extract the required verbatim RFC 9110 normative text or decision criteria.

## 401 (Unauthorized): Normative meaning and headers
INCOMPLETE: The supplied evidence includes only a partial RFC 9110 meaning for 401 (“the request has not been applied because it lacks valid authentication credentials for the target resource”). [S3] However, the supplied evidence does not include the verbatim RFC 9110 subsection text or RFC 9110’s normative MUST/SHOULD language governing whether 401 must include a `WWW-Authenticate` header, nor the required/explicit challenge purpose from the primary specification. [S3]

The only authoritative header-related evidence in the supplied catalog is that RFC 7235 (now obsolete, see RFC 9110) has a section structure including “3.1. 401 Unauthorized” and “4.1. WWW-Authenticate,” but the excerpts provided do not include the actual normative statements about header requirements. [S4]

## 403 (Forbidden): Normative meaning and headers
INCOMPLETE: The supplied evidence does not include any authoritative RFC 9110 normative excerpt that defines the precise meaning of 403 (Forbidden), nor any authoritative RFC-based guidance on response headers for 403. [S3]

The supplied RFC 7235 excerpt similarly provides only a table-of-contents-like structure and does not include normative statements defining 403 or any header requirements/recommendations for 403. [S4]

## Server decision criteria: Choosing 401 vs 403 in real-world scenarios
INCOMPLETE: The supplied evidence does not include the RFC 9110 normative text needed to derive spec-justified decision criteria for when to return 401 versus 403 across the requested cases (unauthenticated vs authenticated-but-not-authorized, missing vs invalid vs expired credentials/tokens, account/role/resource restrictions, and other edge cases). [S3]

Because the authoritative decision logic and the implications/purpose of the 401 challenge are not present in the supplied RFC excerpts, any concrete 401-vs-403 mapping across those scenarios would require non-normative third-party explanations, which is outside the requested evidence basis. [S3][S4]

## Limitations
- The supplied evidence does not include the required verbatim RFC 9110 normative sections/paragraphs for 401 and 403, nor the exact MUST/SHOULD language about `WWW-Authenticate` and 403 behavior. [S3]
- Because the authoritative RFC text needed to derive spec-justified decision criteria is missing, any concrete 401-vs-403 mapping would rely on non-normative third-party explanations, which is outside the requested evidence basis. [S3][S4]

## Sources
- [S3] RFC 9110 - HTTP Semantics - https://httpwg.org/specs/rfc9110.html
- [S4] RFC 7235: Hypertext Transfer Protocol (HTTP/1.1): Authentication | RFC Editor - https://www.rfc-editor.org/info/rfc7235