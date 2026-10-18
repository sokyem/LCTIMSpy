"""SDK-independent core of LCTIMSpy.

Nothing in this package imports ``pyTDFSDK`` at module scope. The Bruker SDK is
loaded on demand by :mod:`LCTIMSpy.core.sdk`, and only extraction needs it;
everything that reads the cache works on a machine without the SDK.
"""
