const init = (fen) => {
  const urlParams = new URLSearchParams(window.location.search);
  const rating = urlParams.get("rating") || 1200;

  var game = new Chess(fen);
  var side = game.turn();

  const onDragStart = (source, piece, position, orientation) => {
    if (game.game_over()) return false;
    if (game.turn() == "w" && piece.search(/^b/) !== -1) return false;
    if (game.turn() == "b" && piece.search(/^w/) !== -1) return false;
  };

  const onDrop = (source, target) => {
    const move = game.move({ from: source, to: target, promotion: "q" });
    if (move === null) return "snapback";
  };

  const onChange = async () => {
    const response = await fetch(`/move/${game.fen()}?rating=${rating}`);
    const evaluation = parseFloat(await response.text());
    $("#cheat").prop("disabled", evaluation > 0);
    if (game.game_over()) {
      alert("Game Over");
      window.location.reload();
    }
    if (game.turn() != side) {
      const response = await fetch(`/maia/${game.fen()}?rating=${rating}`);
      const fen = await response.text();
      game.load(fen);
      board.position(game.fen());
    }
  };

  const config = {
    draggable: true,
    position: game.fen(),
    orientation: game.turn() == "w" ? "white" : "black",
    onDragStart,
    onDrop,
    onChange,
  };
  const board = Chessboard("board", config);

  $("#cheat").on("click", async () => {
    const response = await fetch(`/stockfish/${game.fen()}`);
    const fen = await response.text();
    game.load(fen);
    board.position(game.fen());
  });
};
