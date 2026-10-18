"""Transport the official retrieval tool to the already protected common server."""
import json


def common_results(payload, count, topk):
    rows = payload.get('result')
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError('Common retriever batch dimension changed')
    converted = []
    for docs in rows:
        if not isinstance(docs, list) or len(docs) != topk:
            raise ValueError('Common retriever topk changed')
        out = []
        for item in docs:
            doc = item['document']
            content = doc.get('contents', doc.get('content'))
            if not isinstance(content, str) or not isinstance(item['score'], (int, float)):
                raise ValueError('Common document/score schema changed')
            out.append({'id': doc['id'], 'contents': content, 'score': item['score']})
        converted.append(out)
    return converted


def make_tool(server_url):
    from agent.search.local_retrieval_tool import LocalRetrievalTool, _BatchRequester

    class CommonBatchRequester(_BatchRequester):
        def _process_batch(self, batch):
            if any(task.top_k != 3 for task in batch):
                raise ValueError('Official SearchR1 training fixes retrieval topk3')
            response = self.client.post(self.server_url + '/retrieve', json={
                'queries': [task.query for task in batch], 'topk': 3, 'return_scores': True})
            response.raise_for_status()
            rows = common_results(response.json(), len(batch), 3)
            for task, docs in zip(batch, rows):
                if not task.future.done():
                    task.future.set_result(docs)

    class SharedCommonRetrievalTool(LocalRetrievalTool):
        """Keep official forward/formatting semantics; change only HTTP transport."""

        def _get_or_create_batcher(self):
            server = server_url.rstrip('/')
            key = server + '/branpo-common-api'
            with self._aggregators_lock:
                if key not in self._aggregators:
                    self._aggregators[key] = CommonBatchRequester(server, self.timeout,
                        max_batch_size=self.max_batch_size, batch_interval=self.batch_interval)
                return self._aggregators[key]

    return SharedCommonRetrievalTool
