(function () {
  const SB = (window.SB = window.SB || {});

  SB.storageKeys = {
    sessionId: "switchboard.session_id",
    sidebarCollapsed: "switchboard.sidebar_collapsed",
    feedbackNudgeSeen: "switchboard.feedback.enable_nudge_seen",
  };

  SB.state = SB.state || {
    sessionId: window.localStorage.getItem(SB.storageKeys.sessionId) || null,
    currentModel: "auto",
    composer: {
      isSending: false,
      privateChat: false,
    },
    feedback: {
      enableNudgeShown:
        window.sessionStorage.getItem(SB.storageKeys.feedbackNudgeSeen) === "1",
    },
    openOverlayStack: [],
  };
})();
