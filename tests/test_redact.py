from anchor.redact import redact


def test_aws_access_key():
    text, n = redact("key is AKIAIOSFODNN7EXAMPLE ok")
    assert "AKIAIOSFODNN7EXAMPLE" not in text
    assert "[REDACTED:aws-access-key]" in text
    assert n == 1


def test_github_token():
    text, n = redact("export GH=ghp_abcdefghijklmnopqrstuvwxyz0123456789")
    assert "ghp_" not in text
    assert n >= 1


def test_private_key_block():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n-----END RSA PRIVATE KEY-----"
    text, n = redact(f"found this: {pem} in a screenshot")
    assert "BEGIN RSA PRIVATE KEY" not in text
    assert n == 1


def test_jwt():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    text, n = redact(f"Authorization: {jwt}")
    assert jwt not in text
    assert n >= 1


def test_password_assignment():
    text, n = redact('DB_PASSWORD = "hunter2secret"')
    assert "hunter2secret" not in text
    assert n == 1


def test_bearer_token():
    text, n = redact("curl -H 'Authorization: Bearer sk_live_abcdefghij1234567890xyz'")
    assert "sk_live_abcdefghij1234567890xyz" not in text


def test_clean_text_untouched():
    original = "How do I fix a Django migration conflict on the users table?"
    text, n = redact(original)
    assert text == original
    assert n == 0
