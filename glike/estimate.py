from .glike import *

class Search():
  def __init__(self, x0, bounds = None, precision = 0.05):
    self.names = list(x0.keys())
    self.values = x0.copy()
    if bounds is None:
      bounds = [(0, math.inf) for _ in self.names]
    self.bounds = dict(zip(self.names, bounds))
    self.lrs = {name:0.1 for name in self.names}
    self.precision = precision
  
  def get(self):
    return self.values
  
  def set(self, values):
    self.values = values
  
  def limit(self, name):
    limit = self.bounds[name]
    low = limit[0]; high = limit[1]
    if isinstance(low, str):
      low = eval(low, self.values.copy())
    if isinstance(high, str):
      high = eval(high, self.values.copy())
    return low, high
  
  def up(self, name):
    value = self.values[name]
    lr = self.lrs[name]
    low, high = self.limit(name)
    if value < (low + high)/2:
      step = (value - low) * lr
    else:
      step = (high - value) * lr
    step = max(step, 1e-5)
    values = self.values.copy()
    values[name] = min(high, round(value + step, 5))
    return values
  
  def down(self, name):
    value = self.values[name]
    lr = self.lrs[name]
    low, high = self.limit(name)
    if value < (low + high)/2:
      step = (value - low) * lr
    else:
      step = (high - value) * lr
    step = max(step, 1e-5)
    values = self.values.copy()
    values[name] = round(max(low + 1e-5, value - step), 5)
    return values
  
  def faster(self, name):
    self.lrs[name] = min(0.5, self.lrs[name] * 1.5)
  
  def slower(self, name):
    self.lrs[name] = max(self.precision, self.lrs[name] * 0.5)
  
  def cold(self):
    for name in self.names:
      if self.lrs[name] > self.precision:
        return False
    return True


def maximize(fun, x0, bounds = None, precision = 0.05, epochs = 20, verbose = False):
  # fun: The objective function to be maximized.
  # x0: the dict of initial parameters, such that the initial output would be fun(**x0)
  # bounds: the list of 2-tuples that defines the boundaries
  # precision: a float that defines the (proportional) step size
  # epochs: an integer that defines the maximum number of epochs
  # verbose: True if intermediate results are printed, False if not
  
  search = Search(x0, bounds = bounds, precision = precision)
  names = list(x0.keys())
  
  y0 = fun(**x0)
  print(str(x0) + " " + str(y0), flush = True)
  
  xs = []
  ys = []
  for _ in range(epochs):
    for name in names:
      x = search.get()
      y = fun(**x)
      x_up = search.up(name)
      y_up = fun(**x_up)
      x_down = search.down(name)
      y_down = fun(**x_down)
      
      if verbose:
        print(" ", flush = True)
        print("x_up: " + str(x_up) + " " + str(y_up), flush = True)
        print("x: " + str(x) + " " + str(y), flush = True)
        print("x_down: " + str(x_down) + " " + str(y_down), flush = True)
        print(" ", flush = True)
      
      if (y_up > max(y_down, y)):
        search.set(x_up)
        search.faster(name)
      elif (y_down > max(y_up, y)):
        search.set(x_down)
        search.faster(name)
      else:
        search.slower(name)
    
    x = search.get()
    y = fun(**x)
    xs.append(x); ys.append(y)
    print(str(x) + " " + str(y), flush = True)
    
    if len(ys) >= 5 and sum(ys[-5:-3]) >= sum(ys[-2:]):
      break
  
  idx = ys.index(max(ys))
  x, y = xs[idx], ys[idx]
  return x, y


# =============================================================================
# Reparameterization utilities
# =============================================================================
# These transforms map constrained parameters to unconstrained space,
# making the optimization landscape smoother and more uniform.

def _softplus(x):
  """Numerically stable softplus: log(1 + exp(x))."""
  if x > 20:
    return x
  elif x < -20:
    return math.exp(x)
  else:
    return math.log1p(math.exp(x))

def _inv_softplus(y):
  """Inverse of softplus: log(exp(y) - 1)."""
  if y > 20:
    return y
  elif y < 1e-8:
    return -40.0
  else:
    return math.log(math.expm1(y))

# --- Log transform for positive parameters (population sizes) ---
def _log_transform(value):
  return math.log(value)

def _log_inverse(raw):
  return math.exp(raw)

# --- Logit transform for parameters in (0, 1) (proportions) ---
def _logit_transform(value):
  value = max(1e-10, min(1 - 1e-10, value))
  return math.log(value / (1 - value))

def _logit_inverse(raw):
  return 1.0 / (1.0 + math.exp(-raw))

# --- Ordered transform for ordered times (t1 < t2 < t3 < ...) ---
# Represents ordered times as: t1 = softplus(r1), t2 = t1 + softplus(r2), ...
# Each raw parameter controls the gap to the next time.
def _ordered_transform(times):
  """Convert ordered times to unconstrained raw values."""
  raws = []
  prev = 0.0
  for t in times:
    gap = t - prev
    raws.append(_inv_softplus(gap))
    prev = t
  return raws

def _ordered_inverse(raws):
  """Convert unconstrained raw values back to ordered times."""
  times = []
  cumulative = 0.0
  for r in raws:
    cumulative += _softplus(r)
    times.append(cumulative)
  return times


class ReparamSearch():
  """Coordinate descent optimizer that works in reparameterized (unconstrained) space.
  
  param_types: dict mapping parameter names to their type:
    - "size":       positive values (population sizes), transformed via log
    - "proportion": values in (0,1) (admixture fractions), transformed via logit
    - "time:k":     the k-th time in an ordered sequence (t1 < t2 < t3 ...),
                    transformed via cumulative softplus. k is 0-indexed.
    - "positive":   generic positive values, transformed via log (alias for size)
    - "raw":        no transformation applied (already unconstrained)
  """
  def __init__(self, x0, param_types, precision = 0.05):
    self.names = list(x0.keys())
    self.param_types = param_types
    self.precision = precision
    
    # identify time groups
    self.time_groups = {} # group_id -> list of (name, order_index)
    for name in self.names:
      ptype = param_types.get(name, "raw")
      if ptype.startswith("time"):
        parts = ptype.split(":")
        if len(parts) == 2:
          order = int(parts[1])
        else:
          order = 0
        group = "time"
        self.time_groups.setdefault(group, [])
        self.time_groups[group].append((name, order))
    
    # sort time groups by order
    for group in self.time_groups:
      self.time_groups[group].sort(key = lambda x: x[1])
    
    # transform initial values to raw space
    self.raw_values = {}
    self._to_raw(x0)
    
    self.lrs = {name: 0.1 for name in self.names}
  
  def _to_raw(self, values):
    """Convert constrained values dict to internal raw values."""
    done = set()
    
    # handle time groups first
    for group, members in self.time_groups.items():
      time_names = [name for name, _ in members]
      time_values = [values[name] for name, _ in members]
      raw_times = _ordered_transform(time_values)
      for name, raw in zip(time_names, raw_times):
        self.raw_values[name] = raw
        done.add(name)
    
    # handle remaining parameters
    for name in self.names:
      if name in done:
        continue
      ptype = self.param_types.get(name, "raw")
      value = values[name]
      if ptype in ("size", "positive"):
        self.raw_values[name] = _log_transform(value)
      elif ptype == "proportion":
        self.raw_values[name] = _logit_transform(value)
      else: # "raw"
        self.raw_values[name] = value
  
  def _from_raw(self):
    """Convert internal raw values to constrained values dict."""
    values = {}
    done = set()
    
    # handle time groups
    for group, members in self.time_groups.items():
      time_names = [name for name, _ in members]
      raw_times = [self.raw_values[name] for name in time_names]
      constrained_times = _ordered_inverse(raw_times)
      for name, t in zip(time_names, constrained_times):
        values[name] = round(t, 5)
        done.add(name)
    
    # handle remaining parameters
    for name in self.names:
      if name in done:
        continue
      ptype = self.param_types.get(name, "raw")
      raw = self.raw_values[name]
      if ptype in ("size", "positive"):
        values[name] = max(1e-10, round(_log_inverse(raw), 5))
      elif ptype == "proportion":
        values[name] = min(1 - 1e-5, max(1e-5, round(_logit_inverse(raw), 5)))
      else:
        values[name] = round(raw, 5)
    
    return values
  
  def get(self):
    return self._from_raw()
  
  def set_raw(self, name, raw_value):
    self.raw_values[name] = raw_value
  
  def up(self, name):
    old_raw = self.raw_values[name]
    step = self.lrs[name]
    self.raw_values[name] = old_raw + step
    values = self._from_raw()
    self.raw_values[name] = old_raw
    return values
  
  def down(self, name):
    old_raw = self.raw_values[name]
    step = self.lrs[name]
    self.raw_values[name] = old_raw - step
    values = self._from_raw()
    self.raw_values[name] = old_raw
    return values
  
  def accept_up(self, name):
    step = self.lrs[name]
    self.raw_values[name] += step
  
  def accept_down(self, name):
    step = self.lrs[name]
    self.raw_values[name] -= step
  
  def faster(self, name):
    self.lrs[name] = min(1.0, self.lrs[name] * 1.5)
  
  def slower(self, name):
    self.lrs[name] = max(self.precision, self.lrs[name] * 0.5)
  
  def cold(self):
    for name in self.names:
      if self.lrs[name] > self.precision:
        return False
    return True


def maximize_reparam(fun, x0, param_types, precision = 0.05, epochs = 20, verbose = False):
  """Maximize fun(**x) using coordinate descent in reparameterized space.
  
  This is a drop-in improvement over maximize() that works in transformed
  (unconstrained) parameter space. The reparameterization ensures uniform
  step sizes across parameters of very different scales.
  
  Parameters
  ----------
  fun : callable
      The objective function, called as fun(**x) where x is a dict of
      parameter values in the original (constrained) space.
  x0 : dict
      Initial parameter values in the original space.
  param_types : dict
      Maps parameter names to their type. Supported types:
        "size"         - positive values (e.g. population sizes), log-transformed
        "positive"     - alias for "size"
        "proportion"   - values in (0,1) (e.g. admixture fractions), logit-transformed
        "time:k"       - the k-th element (0-indexed) in an ordered time sequence,
                         transformed via cumulative softplus so that ordering is
                         guaranteed (e.g. {"t1": "time:0", "t2": "time:1", "t3": "time:2"})
        "raw"          - no transformation (default if name not in param_types)
  precision : float
      Minimum step size in raw space before a parameter is considered converged.
  epochs : int
      Maximum number of full sweeps through all parameters.
  verbose : bool
      If True, print intermediate results.
  
  Returns
  -------
  x : dict
      Best parameter values found (in original space).
  y : float
      Maximum likelihood value at x.
  
  Example
  -------
  Using the threeway admixture model from the tutorial::
  
      param_types = {
          "t1": "time:0", "t2": "time:1", "t3": "time:2",
          "r1": "proportion", "r2": "proportion",
          "N": "size", "N_a": "size", "N_b": "size",
          "N_c": "size", "N_d": "size", "N_e": "size"
      }
      x, logp = glike.maximize_reparam(fun, x0, param_types)
  """
  search = ReparamSearch(x0, param_types, precision = precision)
  names = list(x0.keys())
  
  y0 = fun(**x0)
  print(str(x0) + " " + str(y0), flush = True)
  
  xs = []
  ys = []
  for _ in range(epochs):
    for name in names:
      x = search.get()
      y = fun(**x)
      x_up = search.up(name)
      y_up = fun(**x_up)
      x_down = search.down(name)
      y_down = fun(**x_down)
      
      if verbose:
        print(" ", flush = True)
        print("x_up: " + str(x_up) + " " + str(y_up), flush = True)
        print("x: " + str(x) + " " + str(y), flush = True)
        print("x_down: " + str(x_down) + " " + str(y_down), flush = True)
        print(" ", flush = True)
      
      if (y_up > max(y_down, y)):
        search.accept_up(name)
        search.faster(name)
      elif (y_down > max(y_up, y)):
        search.accept_down(name)
        search.faster(name)
      else:
        search.slower(name)
    
    x = search.get()
    y = fun(**x)
    xs.append(x); ys.append(y)
    print(str(x) + " " + str(y), flush = True)
    
    if len(ys) >= 5 and sum(ys[-5:-3]) >= sum(ys[-2:]):
      break
  
  idx = ys.index(max(ys))
  x, y = xs[idx], ys[idx]
  return x, y
