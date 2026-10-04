# HTTP 401 vs 403 (Unauthorized vs Forbidden)

## Answer
- **What does 401 Unauthorized mean?** It means the server **has not applied the request** because the client **lacks valid authentication credentials** for the target resource. [S1]
- **What does 403 Forbidden mean?** It means the server **understood the request** but **refuses to fulfill it**. [S6]
- **When should a server return 401 rather than 403 (and vice versa)?**
  - Return **401 Unauthorized** when the client is **missing credentials or provided invalid/expired credentials**, so the server cannot authenticate the client; a typical rule of thumb is that the client should be able to **authenticate (again) and retry**. [S11][S13]
  - Return **403 Forbidden** when the server has already **recognized/understood the request** (i.e., authentication is not the deciding failure) but the client **is not allowed to access** the resource due to authorization/policy (insufficient privileges). [S6][S14]
  - If **Authorization credentials are already present** and the server still returns **401**, that indicates authorization has been refused for those credentials (i.e., the credential presented was not acceptable for authentication). [S12]

## Limitations
- The decision rules above are supported by secondary guidance and one-part RFC 9110-style definition; the collected results do not include direct RFC 9110 text that fully contrasts 401 vs 403 decision conditions beyond the high-level definitions and a noted `WWW-Authenticate` requirement for 401. [S1][S6]

## Sources
- [S1] 401 responses MUST have a `WWW-Authenticate` header. · Issue #51 · LionC/express-basic-auth · GitHub - https://github.com/LionC/express-basic-auth/issues/51
- [S6] RFC 9110 - HTTP Semantics - https://httpwg.org/specs/rfc9110.html
- [S11] The developer's guide to HTTP error codes - https://workos.com/blog/http-error-codes
- [S12] HTTP Status Codes 401 Unauthorized and 403 Forbidden for ... - https://robertlathanh.com/2012/06/http-status-codes-401-unauthorized-and-403-forbidden-for-authentication-and-authorization-and-oauth
- [S13] HTTP 401 vs 403: Difference, Fixes & When to Use - https://www.authgear.com/post/http-401-vs-403
- [S14] 403 Forbidden vs 401 Unauthorized HTTP responses - https://stackoverflow.com/questions/3297048/403-forbidden-vs-401-unauthorized-http-responses