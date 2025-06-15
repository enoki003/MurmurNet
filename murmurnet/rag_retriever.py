from typing import List, Dict
import libzim
import logging

class RAGRetriever:
    def __init__(self, zim_archive: str):
        self.zim_archive = zim_archive
        self.logger = logging.getLogger(__name__)

    def search_zim(self, query: str, max_results: int = 5) -> List[Dict]:
        """ZIMファイルから検索を行う"""
        try:
            search = libzim.Search(self.zim_archive)
            search.set_query(query)
            
            # get_matches_estimatedが利用できない場合の代替処理
            try:
                estimated_count = search.get_matches_estimated()
            except AttributeError:
                # 代替: 実際の結果を取得して数を確認
                estimated_count = max_results
                self.logger.debug("get_matches_estimated not available, using max_results as estimate")
            
            results = []
            iterator = search.get_results(0, min(estimated_count, max_results))
            
            for i in range(min(estimated_count, max_results)):
                try:
                    entry = iterator.get_entry(i)
                    if entry:
                        results.append({
                            'title': entry.get_title(),
                            'path': entry.get_path(),
                            'content': entry.get_item().get_data().decode('utf-8', errors='ignore')[:1000]
                        })
                except (IndexError, AttributeError) as e:
                    self.logger.debug(f"Entry {i} could not be processed: {e}")
                    break
                    
            return results
            
        except Exception as e:
            self.logger.error(f"ZIM検索エラー: {e}")
            return []