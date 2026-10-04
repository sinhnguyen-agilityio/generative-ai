# HTTP 429 Too Many Requests: RFC, meaning, and retry delay header

## Answer
- **Which RFC defines HTTP 429 Too Many Requests:** **RFC 6585 (Additional HTTP Status Codes)** defines the **429 Too Many Requests** status code. [S6]
- **What HTTP 429 Too Many Requests means:** **429 Too Many Requests** indicates that **the user has sent too many requests in a given amount of time (“rate limiting”)**. [S6]
- **Which response header tells the client how long to wait before retrying:** The response **MAY include a `Retry-After` header** indicating **how long to wait before making a new request**. [S6]

## Sources
- [S6] RFC 6585 - Additional HTTP Status Codes - https://httpwg.org/specs/rfc6585.html