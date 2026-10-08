    def before_actor_update(self, data, outer_step):
        if self.method == 'topd' and self.config.get('topd', {}).get('teacher_backend') == 'local_gpu_fp32_shared_pipeline32b':
            if int(data.meta_info['topd_outer_step']) != int(outer_step):
                raise RuntimeError('Shared32B actor outer identity mismatch')
            from topd32b_shared_pipeline.backend import require_global_ack_before_actor
            require_global_ack_before_actor(self.teacher, int(outer_step))

