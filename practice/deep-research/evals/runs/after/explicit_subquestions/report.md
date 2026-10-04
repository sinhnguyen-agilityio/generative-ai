## Answer
- **RFC that defines HTTP 429 Too Many Requests**: **RFC 6585** (“Additional HTTP Status Codes”), section **4.2** [S5].
- **Meaning of HTTP 429 Too Many Requests**: The server indicates it is **rejecting the request because the client has sent too many requests in a given amount of time** [S1].
- **Response header that tells the client how long to wait before retrying**: The response may include the **`Retry-After`** header, which indicates **how long to wait before making a new request** [S1][S4].

## Sources
- [S1] 429 Too Many Requests - HTTP | MDN - https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/429
- [S4] Retry-After header - HTTP | MDN - https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Retry-After
- [S5] RFC 6585: Additional HTTP Status Codes | RFC Editor - https://www.rfc-editor.org/info/rfc6585