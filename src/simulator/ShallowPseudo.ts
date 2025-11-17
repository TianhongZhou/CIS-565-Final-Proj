/*


Textures needed:

height
previousHeight

fluxX
fluxY
velocityX
velocityY
changeInVelocityX
changeInvelocityY

Constants:
timestep
gridsize


Steps:

computeInitialVelocityX(previousHeight, fluxX, velocityX);
computeInitialVelocityY(previousHeight, fluxY, velocityY);

copyTexture(height -> previousHeight)

shallowHeight(previousHeight, velocityX, velocityY, height, constants);
shallowVelocityXStep1(velocityX, fluxX, previousHeight, changeInVelocityX, constants);
shallowVelocityXStep2(velocityX, fluxY, previousHeight, changeInVelocityX, constants);

shallowVelocityXStep1(velocityY, fluxX, previousHeight, changeInvelocityY, constants);
shallowVelocityXStep2(velocityY, fluxY, previousHeight, changeInvelocityY, constants);

updateVelocityAndFluxX(changeInVelocityX, height, velocity, fluxX, constants);
updateVelocityAndFluxY(changeInVelocityY, height, velocity, fluxY, constants);



TODO: debug shaders, fix texture sampling to account for texture boundary, start research on boundary conditions.
*/