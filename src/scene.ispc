
export void raytrace_ispc(uniform int size, uniform float output[])
{
	varying RNGState* uniform state;
	int seed = programIndex;
	seed_rng(state, seed);

	foreach (i = 0 ... size) {
		output[i] = frandom(state);
    }
}