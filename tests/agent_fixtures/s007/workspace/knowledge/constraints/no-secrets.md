---
id: company/security/no-secrets
type: constraint
scope: global
locked: true
---

# Credentials and secrets

Never commit a credential, password, API key, or other secret to source control. This
includes .env and other configuration files: reference an environment variable or secret
store instead of writing the secret into a file in the repository.
