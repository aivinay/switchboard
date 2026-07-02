(function () {
  const SB = (window.SB = window.SB || {});
  const stack = SB.state.openOverlayStack;

  function remove(handle) {
    const index = stack.indexOf(handle);
    if (index !== -1) {
      stack.splice(index, 1);
    }
  }

  function open(handle) {
    remove(handle);
    stack.push(handle);
  }

  function close(handle, reason = "programmatic") {
    remove(handle);
    handle.close(reason);
  }

  function closeTop(reason = "programmatic") {
    const top = stack.at(-1);
    if (top) {
      close(top, reason);
      return true;
    }
    return false;
  }

  function register(config) {
    return {
      id: config.id,
      element: config.element,
      trigger: config.trigger || null,
      close: config.close,
    };
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && closeTop("escape")) {
      event.preventDefault();
    }
  });

  document.addEventListener("click", (event) => {
    const top = stack.at(-1);
    if (!top) {
      return;
    }
    const target = event.target;
    if (
      top.element.contains(target) ||
      (top.trigger && top.trigger.contains(target))
    ) {
      return;
    }
    close(top, "outside");
  });

  SB.dismissableStack = {
    register,
    open,
    close,
    closeTop,
    remove,
  };
})();
