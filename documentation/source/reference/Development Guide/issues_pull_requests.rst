.. _DevGuideIssuesPullRequests:

Issues and Pull Requests
========================

This page covers our standard contribution workflow, how to open a pull request, how to ask for help, and the most common pitfalls to avoid.

The contribution workflow
-------------------------

For most contributions to Qrisp, we strongly encourage following the **"Issue First, Pull Request Second"** workflow. 

Before you start writing code for a bug fix, a new feature, or a significant documentation change, please open an issue to propose your idea. Treating the issue as a proposal provides several key benefits:

* **It saves time:** Maintainers can review your idea, suggest a specific technical direction, or discuss how it fits into the roadmap before you spend hours implementing it.
* **It prevents duplicate work:** Once an issue is opened and assigned, it signals to the rest of the community that the problem is being handled, preventing conflicting pull requests.
* **It separates the "why" from the "how":** The issue provides a dedicated space to discuss the motivation (e.g., bug reproduction steps or feature logic). The pull request can then focus purely on the code and architecture.

**When to skip opening an issue:**
While the issue-first approach is the standard for most changes, it can be unnecessary for trivial updates. You are welcome to skip opening an issue and go straight to a pull request for:

* Typo fixes in documentation or code comments.
* Fixing broken links.
* Incredibly minor bug fixes where the problem and solution are undisputed.

Asking questions
----------------

There are no stupid questions. If something is unclear, or if you are unsure
whether a change is appropriate, please ask before implementing it. Asking
early almost always saves time.

Good places to reach out:

- **GitHub Issues**: for specific bugs, questions about behaviour, or
  suggestions: https://github.com/eclipse-qrisp/Qrisp/issues
- **Discord**: for broader design conversations:
  https://discord.gg/v5np7DeBaq

Opening a pull request
-----------------------

Before opening a PR, run the test suite for the subsystem(s) you modified. 
For example, if the changes you made affected the functionality in the ``circuit`` module, run:

.. code-block:: bash

    pytest tests/circuit_tests/   # adjust path to match your changes

CI will catch failures, but catching them locally first saves time and
conserves GitHub runner resources.

**Prefer small, focused pull requests.** A PR that does one thing well is
significantly easier to review and gets merged faster. Large refactors should
be split into multiple PRs whenever possible.

Useful references:

- GitHub guide on pull requests: https://docs.github.com/en/pull-requests
- How to write a good commit message: https://cbea.ms/git-commit/

Advertise for your PR!
----------------------

You created a pull request? Great! Let us know about it and make a name for yourself! 

How? Our `Discord <https://discord.gg/fUCFcBAS>`_ is a great starting point - post the link to your PR there and give a quick overview. You will see how helpful and supportive our developers are about your contributions!
Feel free to post about it on social media (LinkedIn).
Additionally, you can write a tutorial about your implementation, granted that it is suited. For a how-to on tutorials check out the :ref:`respective section <DevGuideWriteaTutorial>`! 
Receive your hard-earned recognition and make a name for yourself!


Common mistakes to avoid
-------------------------

These are the most frequent issues in early contributions. Being aware of them
upfront makes the review process smoother for everyone.

**Unintended breaking changes**
  Changes to public APIs require explicit discussion with maintainers. If a
  breaking change is necessary, document it clearly in the PR description.

**Overly large pull requests**
  Large PRs are difficult to review and often delay integration. Keep PRs
  focused on a single issue or improvement.

**Mixing unrelated changes**
  Avoid combining unrelated fixes, style changes, and new features in the same
  PR. Keeping changes cohesive makes review and debugging significantly easier.

**New functionality without tests**
  Any new feature or behaviour change should be accompanied by appropriate
  tests. See :ref:`WritingTests`.

**Ignoring existing project conventions**
  Follow the code structure, naming conventions, and patterns already present
  in the repository unless there is a documented reason to change them.

**Introducing unnecessary dependencies**
  New runtime dependencies should only be added when strictly necessary and
  should be discussed with maintainers first.

**Not running tests before opening the PR**
  Always run ``pytest tests/<subsystem>/`` locally before pushing. CI will
  catch failures, but local validation saves everyone time.
