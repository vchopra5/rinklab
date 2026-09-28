# rinklab

Neural representations research with hockey data.

## xG geometry

Code: [xG/](xG/)

![Comparison of scoring probabilities and gradients](xG/analysis/geometry/scoring_geometry.png)

I was really curious about scoring "geometry": how does the probability of scoring in the NHL vary with shot location?

I started with kernel regression and got a pretty uneven picture. At some points, the gradients point away from the net. I'd expected moving closer to help, at least in front of the goal line.

Other models had smoother gradients. Logistic regression a simpler picture where moving closer to the net improves the probability of scoring. The MLP (2 hidden layers, 128 neurons per layer) picks up more detail, but still smooths over a lot of the little changes you see with the kernel.

The [SIREN paper](https://arxiv.org/abs/2006.09661) showed promising results on fine detail and derivatives, so I tried its sine activations and weight initialization. The picture shows that SIREN's gradients change much more than the MLP's. This makes SIREN and other existing neural representation work an interesting starting point as I build an expected goals model with more than just shot location.