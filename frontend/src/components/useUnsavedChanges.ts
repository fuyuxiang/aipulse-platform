import { Modal } from "antd";
import { useEffect, useRef } from "react";
import { useBlocker } from "react-router-dom";

export function useUnsavedChanges(dirty: boolean): () => void {
  const dirtyRef = useRef(dirty);
  dirtyRef.current = dirty;
  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) =>
      dirtyRef.current && currentLocation.pathname !== nextLocation.pathname,
  );
  useEffect(() => {
    if (blocker.state !== "blocked") return;
    const modal = Modal.confirm({
      title: "当前修改尚未保存",
      content: "离开页面将放弃本次修改。",
      okText: "放弃修改并离开",
      cancelText: "继续编辑",
      onOk: () => blocker.proceed(),
      onCancel: () => blocker.reset(),
    });
    return () => modal.destroy();
  }, [blocker]);
  useEffect(() => {
    const listener = (event: BeforeUnloadEvent) => {
      if (dirtyRef.current) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", listener);
    return () => window.removeEventListener("beforeunload", listener);
  }, []);
  return () => {
    dirtyRef.current = false;
  };
}
