// function initializeDragging() {
//     let selectedSquare = null;
//     let initialX, initialY;
//     const chessboard = document.getElementById('chessboard');

//     if (!chessboard) {
//         console.error("Chessboard element not found.");
//         return; // Exit the function if the chessboard doesn't exist
//     }

//     // Modify the mousedown event listener
//     chessboard.addEventListener('mousedown', function (e) {
//         const clickedElement = e.target;
//         if (clickedElement && clickedElement.hasAttribute('data-square-id')) {
//             selectedSquare = clickedElement.getAttribute('data-square-id');
//             // rest of your drag handling logic
//         }
//     });

//     // Add similar checks and adjustments for mouseup
//     document.addEventListener('mouseup', function (e) {
//         if (selectedSquare) {
//             const droppedElement = e.target;
//             if (droppedElement && droppedElement.hasAttribute('data-square-id')) {
//                 const targetSquareId = droppedElement.getAttribute('data-square-id');
//                 console.log(`Moving piece from ${selectedSquare} to ${targetSquareId}`);
//                 movePiece(selectedSquare, targetSquareId); // Adjust this to send data back to Dash
//                 selectedSquare = null;
//             }
//         }
//     });

//     // Function to simulate dragging the chess piece
//     document.addEventListener('mousemove', function (e) {
//         if (selectedSquare) {
//             const deltaX = e.clientX - initialX;
//             const deltaY = e.clientY - initialY;
//             console.log(`Dragging piece from ${selectedSquare} by (${deltaX}, ${deltaY})`);
//         }
//     });
// }

// function waitForElementToExist(id, callback) {
//     const elem = document.getElementById(id);
//     if (elem) {
//         callback();
//     } else {
//         setTimeout(() => waitForElementToExist(id, callback), 500); // Retry every 500 ms
//     }
// }

// function movePiece(selectedSquare, targetSquareId) {
//     const moveInput = document.getElementById('move-input');
//     moveInput.value = selectedSquare + '-' + targetSquareId;
//     moveInput.dispatchEvent(new Event('change'));
// }

// window.onload = waitForElementToExist('chessboard', initializeDragging);
