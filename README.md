For easilizing neural network design with pytorch with simple design profiles.

For example, using profile string such as:
"1 * 4(R) * 8(R,->,{) * 16(R) * 32(R,+.) * 64(R) * 32(R,->) * 16(R) * 8(R,+.,}) * 4(R) * 8(R,->,{) * 16(R) * 32(R,+.) * 64(R) * 32(R,->) * 16(R) * 8(R,+.,})(31) -- 8(30) -- 8(10) -- 8(1) | 2(S)"
one would be able to build a CNN network with multiple convolution, batchnormalize, dropout, maxpool and resnet layers.

TODO: enable graph plot of network architectures