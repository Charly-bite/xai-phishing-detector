# Contributing to WhatsApp AI Bot

First off, thank you for considering contributing to WhatsApp AI Bot! It's people like you that make WhatsApp AI Bot such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, make one! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

## Fork & create a branch

If this is something you think you can fix, then fork WhatsApp AI Bot and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```
git checkout -b 325-add-new-feature
```

## Implement your fix or feature

At this point, you're ready to make your changes. Feel free to ask for help; everyone is a beginner at first.

## Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with WhatsApp AI Bot's master branch:

```
git remote add upstream https://github.com/Charly-bite/whatsapp-ai-bot.git
git checkout main
git pull upstream main
```

Then update your feature branch from your local copy of master, and push it!

```
git checkout 325-add-new-feature
git rebase main
git push --set-upstream origin 325-add-new-feature
```

Finally, go to GitHub and make a Pull Request.

## Code Style

- Please ensure your code follows the `.editorconfig` rules.
- Run any linting tools before submitting your PR.
