"""Preserve canonical final-answer scoring when actor tokens retain sampler IDs."""

CANONICAL_KEYS = ('canonical_reward_responses', 'canonical_reward_attention_mask',
                  'canonical_reward_info_mask', 'canonical_reward_step_ids')


def reward_response_view(tensors):
    """Return response IDs and their response-only masks; legacy is unchanged."""
    present = [key in tensors for key in CANONICAL_KEYS]
    if any(present) and not all(present):
        raise ValueError('Incomplete canonical reward provenance')
    if all(present):
        response = tensors['canonical_reward_responses']
        width = response.shape[-1]
        attention = (tensors['canonical_reward_attention_mask'][..., -width:] if width
                     else tensors['canonical_reward_attention_mask'][..., :0])
        model_mask = (tensors['canonical_reward_info_mask'][..., -width:] if width
                      else tensors['canonical_reward_info_mask'][..., :0])
        if (attention.shape != response.shape or model_mask.shape != response.shape
                or tensors['canonical_reward_step_ids'].shape != response.shape):
            raise ValueError('Canonical reward response/mask shape mismatch')
        return response, attention, model_mask
    response = tensors['responses']
    width = response.shape[-1]
    attention = tensors['attention_mask'][..., -width:] if width else tensors['attention_mask'][..., :0]
    model_mask = ((tensors['info_mask'][..., -width:] if width else tensors['info_mask'][..., :0])
                  if 'info_mask' in tensors else attention)
    return response, attention, model_mask
