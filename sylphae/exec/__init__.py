"""Execution and verification.

Deliberately does *not* re-export ``verify`` and friends at package level: the function
``sylphae.exec.verify.verify`` and the module ``sylphae.exec.verify`` would then share a name, and
``import sylphae.exec.verify as v`` would silently bind the function instead of the module.
Import from the module path, exactly as INTERFACES.md section 5 names it::

    from sylphae.exec.patch import ApplyResult, apply_to_tree
    from sylphae.exec.verify import VerifyReport, affected_tests, verify
"""
