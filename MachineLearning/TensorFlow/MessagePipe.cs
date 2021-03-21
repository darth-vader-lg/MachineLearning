using Microsoft.ML.Runtime;
using System;

namespace MachineLearning.TensorFlow
{
   internal class MessagePipe<TMessage> : IPipe<TMessage>
   {
      #region Properties
      /// <summary>
      /// La stringa descrittiva del contesto stesso
      /// </summary>
      public string ContextDescription { get; }
      /// <summary>
      /// Indicatore di stato disposed
      /// </summary>
      public bool IsDisposed { get; private set; }
      /// <summary>
      /// Nome del canale
      /// </summary>
      public string Name { get; }
      #endregion
      #region Events
      /// <summary>
      /// Evento disposed del canale
      /// </summary>
      public event EventHandler Disposed;
      /// <summary>
      /// Evento messaggio
      /// </summary>
      public event EventHandler<TMessage> Message;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="name">Nome del canale</param>
      /// <param name="contextDescription">Descrizione del contesto stesso</param>
      public MessagePipe(string name, string contextDescription = null)
      {
         Name = name;
         ContextDescription = contextDescription;
      }
      /// <summary>
      /// Forza il dispose dell'oggetto
      /// </summary>
      public void Dispose()
      {
         try {
            Disposed?.Invoke(this, EventArgs.Empty);
         }
         catch (Exception) { }
         Dispose(disposing: true);
         GC.SuppressFinalize(this);
      }
      /// <summary>
      /// Funzione di dispose
      /// </summary>
      /// <param name="disposing">Se true indica che il dispose dell'oggetto e' stato chiamato da programma</param>
      protected virtual void Dispose(bool disposing)
      {
         if (!IsDisposed)
            IsDisposed = true;
      }
      /// <summary>
      /// Funzione di ricezione di un messaggio
      /// </summary>
      /// <param name="e">Argomenti del messaggio</param>
      protected virtual void OnMessage(TMessage e)
      {
         try {
            Message?.Invoke(this, e);
         }
         catch (Exception) {
         }
      }
      /// <summary>
      /// Processa un'eccezione
      /// </summary>
      /// <typeparam name="TException">Tipo di eccezione</typeparam>
      /// <param name="ex">Exxezione</param>
      /// <returns>L'eccezione</returns>
      public TException Process<TException>(TException ex) where TException : Exception => throw ex;
      /// <summary>
      /// Invia un messaggio
      /// </summary>
      /// <param name="msg">Messaggio</param>
      public void Send(TMessage msg) => OnMessage(msg);
      #endregion
   }
}
