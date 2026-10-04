## Answer

- **401 Unauthorized** means the target resource requires authentication, and the client has not provided credentials (or has provided invalid/insufficient credentials) for that resource [S1].
- **403 Forbidden** means the server understood the request but refused to process it [S10].
- A server should return **401** instead of **403** when the client’s authentication is not adequate or accepted yet, using a `WWW-Authenticate` challenge to indicate the authentication schemes and parameters the client can use for a follow-up request [S1].

## Answer (rule: when to return 401 vs 403)

- A server should return **403** when it receives **valid credentials that are not adequate to gain access** [S6].

## Limitations
- The collected sources do not include the exact RFC 9110 section text for **401 Unauthorized**, so the 401 decision criteria and behavior are supported via the `WWW-Authenticate` challenge expectations rather than a direct RFC 9110 quotation for 401 [S1].

## Sources
- [S1] WWW-Authenticate header - HTTP | MDN - https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/WWW-Authenticate
- [S6] RFC 9110 - HTTP Semantics - https://httpwg.org/specs/rfc9110.html
- [S10] 403 Forbidden - HTTP - MDN Web Docs - Mozilla - https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/403