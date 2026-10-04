## HTTP 401 vs 403: meanings and when to return each

## Answer
- **HTTP 401 Unauthorized means**: the server is using HTTP authentication and responds with `401 Unauthorized` to a request for a protected resource, and the response **must include at least one `WWW-Authenticate` header** containing a **challenge** indicating which authentication scheme(s) can be used and any additional data needed for each scheme. [S1]
- **HTTP 403 Forbidden means**: the server **understood the request** but **refused to authorize it / process it**, i.e., it will not fulfill the request because access is forbidden (a permissions/authorization refusal rather than an authentication challenge). [S8][S6]
- **When a server should return 401 versus 403**: return **401** when the request **requires user authentication** and the client has not provided acceptable authentication credentials for the protected resource (so the client should use the server’s `WWW-Authenticate` challenge and retry with an `Authorization` header). [S1][S2] Return **403** when the server **understood the request** and refuses because the client is not authorized/does not have the required permissions—where re-authentication/changing credentials would not be the protocol-level fix implied by a missing/invalid-authentication condition. [S8][S6]

## Scope
This guidance is limited to what can be established from the provided sources.

## Limitations
- The collected sources do not include verbatim normative RFC 9110 language for 401/403, and instead rely on secondary descriptions (with only partial RFC coverage for 401 via related authentication materials). [S5][S1][S8]
- The collected sources do not provide any explicit specification statement about whether `WWW-Authenticate` is forbidden or merely unnecessary for 403; they only establish the positive `WWW-Authenticate` requirement for 401. [S1][S8]

## Sources
- [S1] WWW-Authenticate header - HTTP | MDN - https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/WWW-Authenticate
- [S2] Getting Error 401--Unauthorized for soap request after added username and password - Help Hub - Postman Community - https://community.postman.com/t/getting-error-401-unauthorized-for-soap-request-after-added-username-and-password/9813
- [S6] Rfc 9110 403 Forbidden Definition - https://www.facebook.com/fb-answers/rfc-9110-403-forbidden-definition?locale=ru_RU
- [S8] 403 Forbidden - HTTP - MDN Web Docs - Mozilla - https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/403
- [S5] RFC 7235: Hypertext Transfer Protocol (HTTP/1.1): Authentication | RFC Editor - https://www.rfc-editor.org/info/rfc7235