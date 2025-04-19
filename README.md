# Variational flow matching based generative models

[Kyunghyun Cho](https://meet.kyunghyuncho.me/)

so, i got curious about [flow matching](https://arxiv.org/abs/2210.02747) after listening to the talk by [Ricky](https://rtqichen.github.io/). i decided to give it a try myself. with Gemini on my side, i had no fear.

i started with a vanilla flow matching based generative model, where i used a vanilla convolutional network. i tested it with MNIST and CelebA. it wasn't as easy as i thought it would be to train these models. perhaps, it is not surprising, since i suspect that a lot of learning updates are wasted as they cancel each other. consider two inputs $x_1$ and $x_2$. we then sample $z_1$ and $z_2$ from a prior distribution. because they are sampled independently from the same, shared distribution, $z_1$ and $z_2$ are exchangeable. two updates that correspond to $(x_1,z_1), (x_2,z_2)$ and $(x_1,z_2), (x_2,z_1)$ will thus cancel each other.

so, i thought; why don't we use an idea of [variational autoencoders](https://arxiv.org/abs/1312.6114)? instead of sampling $z$ from a shared prior, we can sample it from an approximate posterior $q(z|x)$ and make sure this posterior is covered by the shared prior $p(z)$. so, i implemented this portion using another convolutional network. this did help in terms of training, but the resulting models weren't as good as i wanted. 

of course, we are mapping an image to another image, in the case of flow matching for images, and we know that there must be strong structural similarities between the input $(x,t)$ and the predicted velocity vector $v_t$. this suggests using [u-net](https://arxiv.org/abs/1505.04597). with help from Github Copilot, i was able to replace naive convolutional nets with the U-net.

one thing i noticed is that some dataset may have a support that may dramatically differ from the prior $\mathcal{N}(0, 1^2)$. perhaps this makes it hard for the U-nets to learn to capture the velocity field. i thus decided to learn a prior distribution as well by estimating the mean and variance from the data. i could've done it offline, but since i am lazy, i decided to add this as an extra term to the overall loss. this would make the problem easier since the U-nets only need to learn to adjust the shape of the distribution rather than shift it across the space.

even after these, i noticed that my code would work reasonably for some datasets but not for others. then, i was suddenly reminded a remark from [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf), where they talked about how we should consider the discretization as having some positive noise in the input space. this was indeed discussed in the flow matching paper, but obviously i just skipped over it when i implemented the original flow matching loss. i thus revised the code to take into account a small observational variance $\sigma^2=0.01$ when computing the velocity vector.

so, here it is. this code seems to be the minimal working one for **variational flow matching based generative models**.

i should write a full blog post at some point ... but i'm lazy, and the sun is too good outside!

p.s. of course, i was working on this little by little here and there whenever i had time. with such a lack of focus, i made a ton of mistakes here and there and also were misled by Gemini and Claude non-stop here and there. only if i could list up all those dumb things i tried ... 
