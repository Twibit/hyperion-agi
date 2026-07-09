use async_trait::async_trait;
use crate::cortex::MotorCommand;
use crate::error::{TkError, TkResult};
use crate::intent::{Intent, Outcome};
use super::Limb;

/// A biological effector for manipulating remote network sockets (web requests, APIs).
pub struct NetLimb {
    client: reqwest::Client,
}

impl NetLimb {
    /// Grow a new network limb.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Limb for NetLimb {
    fn name(&self) -> &str {
        "NetLimb"
    }

    async fn execute(&self, cmd: MotorCommand) -> TkResult<Outcome> {
        match cmd {
            MotorCommand::Fetch { method, url, payload } => {
                let mut req = match method.to_uppercase().as_str() {
                    "GET" => self.client.get(&url),
                    "POST" => self.client.post(&url),
                    "PUT" => self.client.put(&url),
                    "DELETE" => self.client.delete(&url),
                    _ => return Err(TkError::Other(anyhow::anyhow!("Unsupported method: {}", method))),
                };

                if let Some(p) = payload {
                    req = req.header("Content-Type", "application/json").body(p);
                }

                let response = req.send().await
                    .map_err(|e| TkError::Other(anyhow::anyhow!("Network error: {}", e)))?;

                if !response.status().is_success() {
                    return Err(TkError::Other(anyhow::anyhow!("HTTP Error: {}", response.status())));
                }

                let bytes = response.bytes().await
                    .map_err(|e| TkError::Other(anyhow::anyhow!("Failed to read body: {}", e)))?;

                Ok(Outcome::NetworkResponse { data: bytes.to_vec() })
            },
            _ => Err(TkError::Other(anyhow::anyhow!("NetLimb cannot execute {:?}", cmd))),
        }
    }
}
