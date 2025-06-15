class ModuleAdapters:
    # ...existing code...
    
    def sync_communication_system(self):
        """通信システムの同期処理"""
        try:
            # get_all_storage_dataメソッドが存在しない場合の代替処理
            if hasattr(self.communication_manager, 'get_all_storage_data'):
                storage_data = self.communication_manager.get_all_storage_data()
            elif hasattr(self.communication_manager, 'get_storage_data'):
                storage_data = self.communication_manager.get_storage_data()
            elif hasattr(self.communication_manager, 'storage'):
                storage_data = getattr(self.communication_manager.storage, 'data', {})
            else:
                # フォールバック: 空のデータで継続
                storage_data = {}
                self.logger.warning("通信マネージャーからストレージデータを取得できませんでした")
            
            # 各モジュールに同期データを配布
            for module_name in self.module_names:
                if hasattr(self, f'{module_name}_bridge'):
                    bridge = getattr(self, f'{module_name}_bridge')
                    if hasattr(bridge, 'sync_data'):
                        bridge.sync_data(storage_data)
                        
        except Exception as e:
            self.logger.error(f"通信システム同期エラー: {e}")
            # エラーが発生しても処理を継続
            pass
    
    # ...existing code...