# verus-colocation
Formally verified collocation dictionary builder in Rust + Verus.

POS-tags a raw corpus via spaCy (through Rust-Python bindings), extracts collocations grouped by syntactic pattern, ranks by frequency, and outputs structured dictionary entries.

## Verification

Requires [Verus](https://github.com/verus-lang/verus) built in `../verus/`.

```bash
./scripts/check.sh --require-verus --forbid-trusted-escapes --min-verified 12
```

## License

MIT
