## Answer
- **RFC defining HTTP 429 Too Many Requests:** HTTP status **429 Too Many Requests** is defined in **RFC 6585, section 7.2 ("429 Too Many Requests")** [S4].
- **Meaning of HTTP 429 Too Many Requests:** It means the client has sent **too many requests in a given amount of time**, and the server is effectively **rate limiting** by asking the client to slow down [S1].
- **Response header for retry timing:** The response header **`Retry-After`** tells the client **how long to wait before making a new request** after receiving 429 [S1][S2].

## Sources
- [S1] 429 Too Many Requests - HTTP | MDN - https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/429
- [S2] Retry-After header - HTTP | MDN - https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Retry-After
- [S4] RFC 6585: Additional HTTP Status Codes | RFC Editor - https://www.rfc-editor.org/info/rfc6585