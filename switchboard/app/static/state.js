(function () {
  const SB = (window.SB = window.SB || {});

  SB.storageKeys = {
    sessionId: "switchboard.session_id",
    sidebarCollapsed: "switchboard.sidebar_collapsed",
    privateChat: "switchboard.private_chat",
    feedbackNudgeSeen: "switchboard.feedback.enable_nudge_seen",
  };

  SB.state = SB.state || {
    sessionId: window.localStorage.getItem(SB.storageKeys.sessionId) || null,
    currentModel: "auto",
    composer: {
      isSending: false,
      privateChat: window.localStorage.getItem(SB.storageKeys.privateChat) === "1",
    },
    sidebar: {
      collapsed: window.localStorage.getItem(SB.storageKeys.sidebarCollapsed) === "1",
    },
    feedback: {
      enableNudgeShown:
        window.sessionStorage.getItem(SB.storageKeys.feedbackNudgeSeen) === "1",
    },
    openOverlayStack: [],
  };
})();
