# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainer directly or use GitHub private vulnerability reporting
3. Include a detailed description of the vulnerability and steps to reproduce

## Security Best Practices

When deploying PhantomBot:

- **Never commit secrets** — Use `.env` files and environment variables
- **Use the example configs** — Copy `config.example.json` to `config.json` and fill in your values
- **Restrict API keys** — On your exchange, enable only trading permissions (disable withdrawals) and whitelist your server IP
- **Run behind a reverse proxy** — If exposing the dashboard, use nginx/Caddy with TLS
- **Keep dependencies updated** — Regularly update Freqtrade and Python packages
- **Use paper trading first** — Always test with `dry_run: true` before using real funds
- **Monitor the circuit breaker** — The bot halts at 20% drawdown, but always monitor your positions

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
