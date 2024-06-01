from typing import Any, List, Dict, Set, Callable

class Info:
    def __init__(self, name:str, alias:Set[str], **kwargs) -> None:
        self.name = name
        self.alias = alias
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    def __eq__(self, item) -> bool:
        if isinstance(item, str):
            return self.name == item or item in self.alias
        if isinstance(item, Info):
            return self.name == item.name and self.alias == item.alias
    
    def __repr__(self):
        return f"Info(name={self.name}, alias={str(self.alias)})"
    
class InfoList:
    def __init__(self, info_list:List[Info]) -> None:
        self.info_list:List[Info] = info_list
    
    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        if isinstance(index, str):
            results = []
            for i in self.info_list:
                if i == index:
                    results.append(i)
            if len(results) == 1:
                results = results[0]
        if isinstance(index, int):
            results = self.info_list[index]
        if isinstance(index, slice):
            results = self.info_list[index]
        return results
    
    def __contains__(self, item):
        if isinstance(item, str):
            return bool(self.__getitem__(item))
        if isinstance(item, Info):
            return item in self.info_list
    
    def __repr__(self):
        return f"InfoList{str([info.name for info in self.info_list])}"