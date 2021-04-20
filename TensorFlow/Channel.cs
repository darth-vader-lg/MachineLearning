using Microsoft.ML.Runtime;
using System;

namespace MachineLearning.TensorFlow
{
   /// <summary>
   /// Canale di messaggistica dei contesti TensorFlow
   /// </summary>
   internal class Channel : IChannel
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
      /// Evento di log di un messaggio
      /// </summary>
      public event EventHandler<LoggingEventArgs> Log;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="name">Nome del canale</param>
      /// <param name="contextDescription">Descrizione del contesto stesso</param>
      public Channel(string name, string contextDescription = null)
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
      /// Emette un messaggio di errore
      /// </summary>
      /// <param name="sensitivity">Sensitivita'</param>
      /// <param name="fmt">Formato messaggio</param>
      public void Error(MessageSensitivity sensitivity, string fmt) =>
         OnLog(new LoggingEventArgs(fmt, ChannelMessageKind.Error, Name));
      /// <summary>
      /// Emette un messaggio di errore
      /// </summary>
      /// <param name="sensitivity">Sensitivita'</param>
      /// <param name="fmt">Formato messaggio</param>
      /// <param name="args">Argomenti messaggio</param>
      public void Error(MessageSensitivity sensitivity, string fmt, params object[] args) =>
         OnLog(new LoggingEventArgs(string.Format(fmt, args), ChannelMessageKind.Error, Name));
      /// <summary>
      /// Emette un messaggio di informazione
      /// </summary>
      /// <param name="sensitivity">Sensitivita'</param>
      /// <param name="fmt">Formato messaggio</param>
      public void Info(MessageSensitivity sensitivity, string fmt) =>
         OnLog(new LoggingEventArgs(fmt, ChannelMessageKind.Info, Name));
      /// <summary>
      /// Emette un messaggio di informazione
      /// </summary>
      /// <param name="sensitivity">Sensitivita'</param>
      /// <param name="fmt">Formato messaggio</param>
      /// <param name="args">Argomenti messaggio</param>
      public void Info(MessageSensitivity sensitivity, string fmt, params object[] args) =>
         OnLog(new LoggingEventArgs(string.Format(fmt, args), ChannelMessageKind.Info, Name));
      /// <summary>
      /// Funzione di log di un messaggio
      /// </summary>
      /// <param name="e">Argomenti del messaggio</param>
      protected virtual void OnLog(LoggingEventArgs e)
      {
         try {
            Log?.Invoke(this, e);
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
      /// Emette un messaggio
      /// </summary>
      /// <param name="msg">Il messaggio'</param>
      public void Send(ChannelMessage msg) =>
         OnLog(new LoggingEventArgs(msg.Message, msg.Kind, Name));
      /// <summary>
      /// Emette un messaggio di trace
      /// </summary>
      /// <param name="sensitivity">Sensitivita'</param>
      /// <param name="fmt">Formato messaggio</param>
      public void Trace(MessageSensitivity sensitivity, string fmt) =>
         OnLog(new LoggingEventArgs(fmt, ChannelMessageKind.Trace, Name));
      /// <summary>
      /// Emette un messaggio di trace
      /// </summary>
      /// <param name="sensitivity">Sensitivita'</param>
      /// <param name="fmt">Formato messaggio</param>
      /// <param name="args">Argomenti messaggio</param>
      public void Trace(MessageSensitivity sensitivity, string fmt, params object[] args) =>
         OnLog(new LoggingEventArgs(string.Format(fmt, args), ChannelMessageKind.Trace, Name));
      /// <summary>
      /// Emette un messaggio di warning
      /// </summary>
      /// <param name="sensitivity">Sensitivita'</param>
      /// <param name="fmt">Formato messaggio</param>
      public void Warning(MessageSensitivity sensitivity, string fmt) =>
         OnLog(new LoggingEventArgs(fmt, ChannelMessageKind.Warning, Name));
      /// <summary>
      /// Emette un messaggio di warning
      /// </summary>
      /// <param name="sensitivity">Sensitivita'</param>
      /// <param name="fmt">Formato messaggio</param>
      /// <param name="args">Argomenti messaggio</param>
      public void Warning(MessageSensitivity sensitivity, string fmt, params object[] args) =>
         OnLog(new LoggingEventArgs(string.Format(fmt, args), ChannelMessageKind.Warning, Name));
      #endregion
   }
}
