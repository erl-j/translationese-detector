"""Rule families. Each module tsw/rules/<family>.py exposes

    RULES = [...]            list of rule objects (see tsw/rules/base.py)
    FAMILY = "<family>"      optional; defaults to the module name
    FAMILY_CAP = 4.0         optional; cap on the family's summed contribution

and is picked up automatically by `discover()` (modules named base or
starting with "_" are skipped). There is deliberately NO central registry:
adding a family = adding one file.
"""
import importlib
import pkgutil

SKIP = {"base"}

_cache = None


def discover(reload=False):
    """Import every family module. Returns (rules, family_caps, modules)."""
    global _cache
    if _cache is not None and not reload:
        return _cache
    rules, caps, modules, seen = [], {}, {}, {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name in SKIP or info.name.startswith("_") or info.ispkg:
            continue
        mod = importlib.import_module("%s.%s" % (__name__, info.name))
        mod_rules = getattr(mod, "RULES", None)
        if mod_rules is None:
            continue
        family = getattr(mod, "FAMILY", info.name)
        if hasattr(mod, "FAMILY_CAP"):
            caps[family] = float(mod.FAMILY_CAP)
        modules[family] = mod.__name__
        for r in mod_rules:
            if r.family is None:
                r.family = family
            if r.name in seen:
                raise ValueError("duplicate rule name %r in %s and %s" % (r.name, seen[r.name], mod.__name__))
            seen[r.name] = mod.__name__
            rules.append(r)
    _cache = (rules, caps, modules)
    return _cache
