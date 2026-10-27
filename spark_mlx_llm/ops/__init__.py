"""Operator-local configuration, dispatch and implementations.

The model calls each operator's dispatch_<operator>() entry. Policy is declared
beside the operator, sharing OperatorConfig validation. LinearConfig is also
used by QKV/FFN; automatic composite operators need no extra flags.
"""

__all__ = ["attention_output", "base", "ffn", "linear", "qkv", "sdpa"]
