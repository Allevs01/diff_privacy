

import opendp.prelude as dp
dp.enable_features('contrib')
laplace_mechanism = dp.space_of(float) >> dp.m.then_laplace(scale=1.)
dp_value = laplace_mechanism(123.0)
print(dp_value)

