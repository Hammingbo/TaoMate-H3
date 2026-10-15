"""Reuse the request's coupled Gaussian pair at a Progressive resolution lift."""


def lift_with_request_noise(rows, *, low_noise_rows, high_noise_rows,
                            temporal_length, temporal_offset, source_canvas,
                            target_canvas, video_sigma, noise_seed):
    from digitalhuman_h3.progressive_resolution import (
        PROGRESSIVE_LIFT_SCHEMA, PROGRESSIVE_SWITCH_STATE_INDEX, resize_video_state_rows,
    )

    resize = dict(temporal_length=temporal_length, source_canvas=source_canvas,
                  target_canvas=target_canvas)
    upsampled = resize_video_state_rows(rows, **resize)
    low_noise_upsampled = resize_video_state_rows(low_noise_rows, **resize)
    residual = (high_noise_rows - low_noise_upsampled).to(upsampled.device)
    lifted = upsampled + float(video_sigma) * residual
    return lifted, {
        "schema": PROGRESSIVE_LIFT_SCHEMA,
        "state_index": PROGRESSIVE_SWITCH_STATE_INDEX,
        "source_canvas": source_canvas.as_dict(),
        "target_canvas": target_canvas.as_dict(),
        "video_sigma": float(video_sigma), "noise_seed": int(noise_seed),
        "temporal_offset": int(temporal_offset),
        "noise_indexing": "per_temporal_latent_unit",
        "spatial_interpolation": "bilinear_align_corners_false",
        "high_frequency_residual": "native_high_noise_minus_bilinear_projected_low_noise",
        "state_variance_policy": "coupled_native_high_noise_reconstruction",
        "initial_noise_coupling": {
            "schema": "digitalhuman_h3.progressive_noise_coupling.v1",
            "policy": "separable_orthonormal_lowpass_projection",
            "noise_seed": int(noise_seed), "temporal_offset": int(temporal_offset),
            "temporal_length": int(temporal_length),
            "low_marginal": "iid_standard_gaussian", "high_marginal": "iid_standard_gaussian",
            "shared_random_source": True,
        },
        "audio_transform": "identity", "reused_request_noise": True,
    }
